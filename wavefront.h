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

#ifndef WAVEFRONT_H
#define WAVEFRONT_H

#include <cuda.h>
#include "cuda_helpers.cuh"

#define SEQ_TYPE char

// Limit to 8GiB to be safe
#define MAX_GPU_SIZE (1L << 33)

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS 2147483647

typedef int16_t ewf_offset_t;

struct edit_wavefront_t {
    int hi;
    int lo;
    ewf_offset_t* offsets;
};

struct edit_wavefronts_t {
    edit_wavefront_t wavefront;
    edit_wavefront_t next_wavefront;
};

struct WF_element {
    SEQ_TYPE* text;
    SEQ_TYPE* pattern;
    // Asume len(pattern) == len(text)
    size_t len;
};

class Sequences {
public:
    // Array of WF_element type containing all the elements to be computed.
    WF_element* elements;
    // Same but address on GPU
    WF_element* d_elements;
    // Length of the previous arrays (number of sequences to compute in parallel)
    size_t num_elements;

    size_t max_distance;
    size_t sequence_len;

    // Wavefront information on GPU, offsets are initialized with -1 to avoid
    // loop peeling on the compute kernel.
    edit_wavefronts_t* d_wavefronts;

    Sequences (WF_element* e, size_t num_e, size_t seq_len) : \
                                                elements(e), \
                                                num_elements(num_e), \
                                                sequence_len(seq_len), \
                                                max_distance(seq_len * 2) {}

    bool GPU_memory_init ();
    bool GPU_memory_free ();
    void GPU_launch_wavefront_distance ();
};

#endif // Header guard WAVEFRONT_H
