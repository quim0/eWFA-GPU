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
#include <stdio.h>
#include <string.h>
#include "cuda_helpers.cuh"

#define SEQ_TYPE char

// Limit to 8GiB to be safe
#define MAX_GPU_SIZE (1L << 33)

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS 2147483647

typedef int16_t ewf_offset_t;
typedef char edit_cigar_t;

class Cigars {
public:
    __host__ Cigars (int n, size_t max_d, bool device) {
        // +1 to store a nullbyte
        this->max_distance = max_d + 1;
        this->num_cigars = n;
        if (device) {
            cudaMalloc((void**) &(this->data), this->cigars_size_bytes());
            CUDA_CHECK_ERR
            cudaMemset(this->data, 0, this->cigars_size_bytes());
            CUDA_CHECK_ERR
        }
        else {
            this->data = (edit_cigar_t*)calloc(this->cigars_size_bytes(), 1);
            if (!this->data) {
                fprintf(stderr, "Out of memory on host. (%s:%d)", __FILE__, __LINE__);
                exit(1);
            }
        }
    }

    __host__ __device__ edit_cigar_t* get_cigar (int n) const {
        return &this->data[n * this->max_distance];
    }

    __host__ void print_cigar (int n) {
        // Cigar is in reverse order
        edit_cigar_t* curr_cigar = this->get_cigar(n);
        size_t cigar_len = strnlen(curr_cigar, this->max_distance);
        for (int i=(cigar_len - 1); i >= 0; i--) {
            printf("%c", curr_cigar[i]);
        }
        printf("\n");
    }

    __host__ void copyIn (Cigars device_cigars) {
        cudaMemcpy(this->data, device_cigars.data, this->cigars_size_bytes(), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERR;
    }
private:
    size_t cigars_size_bytes () {
        return this->num_cigars * this->max_distance * sizeof(edit_cigar_t);
    }
    size_t max_distance;
    int num_cigars;
    edit_cigar_t *data;
};

#define OFFSETS_TOTAL_ELEMENTS(max_d) (max_d * max_d + (max_d * 2 + 1))
#define OFFSETS_PTR(offsets_mem, d) (offsets_mem + ((d)*(d)) + (d))

struct edit_wavefronts_t {
    uint32_t d;
    ewf_offset_t* offsets_base;
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
    size_t batch_size;
    // Current batch index
    int batch_idx;

    // Pointer of wavefronts on device
    edit_wavefronts_t* d_wavefronts;
    // Array of wavefronts on host for backtrace calculation
    edit_wavefronts_t* wavefronts;

    Cigars d_cigars;
    Cigars h_cigars;

    Sequences (WF_element* e, size_t num_e, size_t seq_len, size_t batch_size) : \
                                                elements(e),               \
                                                num_elements(num_e),       \
                                                sequence_len(seq_len),     \
    // TODO: Assume error of max 20%, arbitrary chose
                                                max_distance(seq_len / 5), \
                                                batch_size(batch_size),    \
                                                batch_idx(0),              \
                                                d_cigars(
                                                    Cigars(batch_size,
                                                           seq_len * 2,
                                                           true)),         \
                                                h_cigars(
                                                    Cigars(batch_size,
                                                           seq_len * 2,
                                                           false)) {
        // Initialize CPU memory
        this->offsets_host_ptr = (ewf_offset_t*)calloc(
                            OFFSETS_TOTAL_ELEMENTS(this->max_distance) * this->batch_size,
                            sizeof(ewf_offset_t));
        if (!this->offsets_host_ptr) {
            // TODO: Reorganize code so we can include WF_FATAL here
            fprintf(stderr, "Out of memory on CPU. (%s:%d)", __FILE__, __LINE__);
            exit(1);
        }

        this->wavefronts = (edit_wavefronts_t*)calloc(this->batch_size,
                                                      sizeof(edit_wavefronts_t));
        if (!this->wavefronts) {
            fprintf(stderr, "Out of memory on CPU. (%s:%d)", __FILE__, __LINE__);
            exit(1);
        }
    }
    bool GPU_memory_init ();
    bool GPU_memory_free ();
    bool GPU_prepare_memory_next_batch ();
    void backtrace ();
    void GPU_launch_wavefront_distance ();

private:
    SEQ_TYPE* sequences_device_ptr;
    ewf_offset_t* offsets_device_ptr;
    ewf_offset_t* offsets_host_ptr;
};

#endif // Header guard WAVEFRONT_H
