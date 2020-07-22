/*
 * Copyright (c) 2020 Quim Aguado
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "utils.h"

void SequenceReader::initialize_sequences () {
    // Allocate memory for the array that will contain all the elements to
    // process.
    DEBUG("Trying to allocate memory to store all the sequences. (%zu bytes)",
          this->batch_size * sizeof(WF_element)
          );
    this->sequences = (WF_element*)calloc(this->batch_size, sizeof(WF_element));
    if (this->sequences == NULL)
        WF_FATAL(NOMEM_ERR_STR);
    DEBUG("Sequences memory allocated (%p)", this->sequences);
}

size_t SequenceReader::sequence_buffer_size () {
    // Sequence length
    //     +1 for the final nullbyte
    //     +1 for the initial '<' or '>' character
    //     * 2 because max sequence length
    return this->seq_len * 2 + 2;
}

bool SequenceReader::skip_n_alignments (int n) {
    size_t buf_size = this->sequence_buffer_size();
    SEQ_TYPE* buf = this->create_sequence_buffer();

    if (!fp)
        this->fp = fopen(this->seq_file, "r");
    if (!fp) {
        WF_ERROR("Could not open the file %s\n", this->seq_file);
        return false;
    }

    int idx = 0;
    int num_seqs_to_skip = n * 2;

    while (idx++ < num_seqs_to_skip && getline(&buf, &buf_size, this->fp) > 0);

    free(buf);

    if (idx < num_seqs_to_skip) {
        WF_ERROR("Could not skip enough sequences.");
        return false;
    }

    return true;
}

#if 0
void SequenceReader::create_sequences_buffer () {
    int bytes_to_alloc = this->batch_size * this->seq_len * 2 * sizeof(SEQ_TYPE);
    DEBUG("Trying to allocate %d pinned memory MiB to store the sequences.",
          bytes_to_alloc / (1 << 20));
    cudaMallocHost((void**)&this->sequences_mem, bytes_to_alloc);
    if (!this->sequences_mem)
        WF_FATAL(NOMEM_ERR_STR);
    DEBUG("Allocated %d pinned memory MiB to store the sequences.",
          bytes_to_alloc / (1 << 20));
}
#endif

SEQ_TYPE* SequenceReader::get_sequences_buffer () const {
#if 0
    if (this->sequences_mem == NULL) {
        this->create_sequences_buffer();
    }
#endif
    return this->sequences_mem;
    
}

SEQ_TYPE* SequenceReader::create_sequence_buffer () {
    // +1 to include the nullbyte
    SEQ_TYPE* buf = (SEQ_TYPE*)calloc(this->sequence_buffer_size(), sizeof(SEQ_TYPE));
    if (buf == NULL) {
        WF_FATAL(NOMEM_ERR_STR);
    }
    return buf;
}

bool SequenceReader::read_n_sequences (int n) {
    if (this->seq_len == 0) {
        DEBUG("Sequence length specified is 0, nothing to do.");
        return true;
    }

    if (n > this->batch_size) {
        WF_ERROR("Number of alignments to read from file \"%s\"can not be "
                 "bigger than the batch size (%zu)", this->seq_file,
                 this->batch_size);
        return false;
    }

    if (!fp)
        this->fp = fopen(this->seq_file, "r");
    if (!fp) {
        WF_ERROR("Could not open the file %s\n", this->seq_file);
        return false;
    }

    DEBUG("Starting to read the file for sequences.")

    if (this->sequences == NULL) this->initialize_sequences();

    // Buffer to temporary allocate each line
    size_t buf_size = this->sequence_buffer_size();
    SEQ_TYPE* buf = this->create_sequence_buffer();

    // Big chunk of memory where all the sequences will be stored
    // The final nullbyte WILL NOT be stored as we already know the string size
    // num_sequences is the number of sequences pairs (pattern/text), so it's
    // nedded to double the space required.
    SEQ_TYPE* seq_alloc = this->get_sequences_buffer();

    bool retval = true;

    uint32_t idx = 0;
    int sequence_idx = 0;
    int elem_idx = 0;
    // Sequence pair format in file:
    // >TEXTGGG
    // <PATTERN
    ssize_t length;
    while (sequence_idx < n && (length = getline(&buf, &buf_size, this->fp)) > 0) {
        WF_element *curr_elem = &(this->sequences[sequence_idx]);
        // Pointer where the current sequence will be allocated in the big
        // memory chunck
        SEQ_TYPE* curr_seq_ptr = this->get_sequences_buffer() + idx * this->max_seq_len;

        if (elem_idx == 0) {
            // First element of the sequence (TEXT)
            if (buf[0] != '>') {
                WF_ERROR("Invalid sequence format on line %d.\n"
                         "    %s\n", idx, buf);
                // TODO: check if all used memory is ceaned up
                retval = false;
                break;
            }
            // +1 to avoid the '>' character
            memcpy(curr_seq_ptr, buf + 1, length - 2);
            curr_elem->text = curr_seq_ptr;
            curr_elem->tlen = length - 2; // -1 for the initial >, -1 for \n
        } else if (elem_idx == 1) {
            // Second element of the sequence (PATTERN)
            if (buf[0] != '<') {
                WF_ERROR("Invalid sequence format on line %d.\n"
                         "    %s\n", idx, buf);
                // TODO: check if all used memory is ceaned up
                retval = false;
                break;
            }
            // +1 to avoid the '<' character
            memcpy(curr_seq_ptr, buf + 1, length - 2);
            curr_elem->pattern= curr_seq_ptr;
            curr_elem->plen = length - 2; // -1 for <, -1 for \n
        }

        idx++;
        // Current sequence pair
        sequence_idx = idx / 2;
        // First or second element of the sequence pair
        elem_idx = idx % 2;

        this->num_sequences_read++;
    }

#ifdef DEBUG_MODE
    if (retval) {
        DEBUG("File read and sequences stored.\n"
              "    First sequece: %.5s...", this->sequences[0].text);
    } else
        DEBUG("There were some errors while processing the file.");
#endif
    free(buf);
    return retval;
}

void SequenceReader::destroy () {
    cudaFree(sequences_mem);
    this->sequences_mem = NULL;
    free(sequences);
    this->sequences = NULL;
}
