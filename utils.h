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

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wavefront_structures.h"
#include "logger.h"

// TODO: Method to free the sequences memory
class SequenceReader {
public:
    char *seq_file;
    // Length of each sequence (all sequences must have the same number of
    // elements in this version)
    // TODO: All elements always have the same number of elements?
    size_t seq_len;
    // Number of sequences pairs in the "sequences" list and in the file
    size_t num_sequences;

    FILE *fp;
    // Chunk of memory where all the sequences will be stored, the final
    // nullbytes doesn't need to be stored as we already know the sequence
    // length.
    SEQ_TYPE* sequences_mem;
    size_t batch_size;
    int num_sequences_read;

    // Array that will be filled with the sequences from the file
    WF_element* sequences;

    SequenceReader (char* seq_file, size_t seq_len, size_t num_sequences,
                                    size_t batch_size) :                    \
                                              seq_file(seq_file),           \
                                              seq_len(seq_len),             \
                                              num_sequences(num_sequences), \
                                              batch_size(batch_size),       \
                                              sequences(NULL),              \
                                              sequences_mem(NULL),          \
                                              fp(NULL),                     \
                                              num_sequences_read(0) {
        DEBUG("SequenceReader created:\n"
              "    File: %s\n"
              "    Sequence length: %zu\n"
              "    Number of sequences: %zu", seq_file, seq_len, num_sequences);
    }

    bool skip_n_alignments (int n);
    bool read_n_sequences (int n);
    bool read_batch_sequences () {
        return read_n_sequences(this->batch_size);
    }
    void destroy ();

private:
    void initialize_sequences ();
    size_t sequence_buffer_size ();
    void create_sequences_buffer ();
    SEQ_TYPE* get_sequences_buffer ();
    SEQ_TYPE* create_sequence_buffer ();
};

#endif // Header guard UTLS_H
