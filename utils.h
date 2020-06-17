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
#include "wavefront.cuh"

#ifdef DEBUG_MODE
#define DEBUG(...) {\
    char tmp[512];\
    snprintf(tmp, 512, __VA_ARGS__); \
    fprintf(stderr, "DEBUG: %s (%s:%d)\n", tmp, __FILE__, __LINE__); \
    }
#else
#define DEBUG(fmt, ...)
#endif

#define NOMEM_ERR_STR "Could not allocate memory.\n"

#define WF_ERROR(...) fprintf(stderr, __VA_ARGS__)

// TODO: Handle error before exiting? (free gpu memory?)
#define WF_FATAL(...) { \
    WF_ERROR(__VA_ARGS__); fflush(stdout); fflush(stderr); exit(1); \
    }

class SequenceReader {
public:
    char *seq_file;
    // Length of each sequence (all sequences must have the same number of
    // elements in this version)
    // TODO: All elements always have the same number of elements?
    size_t seq_len;
    // Number of sequences pairs in the "sequences" list and in the file
    size_t num_sequences;

    // Array that will be filled with the sequences from the file
    WF_element* sequences;

    SequenceReader (char* seq_file, size_t seq_len, size_t num_sequences) : \
                                              seq_file(seq_file),           \
                                              seq_len(seq_len),             \
                                              num_sequences(num_sequences), \
                                              sequences(NULL) {
        DEBUG("SequenceReader created:\n"
              "    File: %s\n"
              "    Sequence length: %zu\n"
              "    Number of sequences: %zu", seq_file, seq_len, num_sequences);
    }

    bool read_sequences ();

private:
    void initialize_sequences ();
    void free_sequences ();
    size_t sequence_buffer_size ();
    SEQ_TYPE* create_sequence_buffer ();
};

#endif // Header guard UTLS_H
