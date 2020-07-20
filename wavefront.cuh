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
#include "wavefront_structures.h"
#include "logger.h"
#include "utils.h"

// Limit to 8GiB to be safe
#define MAX_GPU_SIZE (1L << 33)

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS 2147483647

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

    __host__ __device__ edit_cigar_t* get_cigar (const int n) const {
        return &this->data[n * this->max_distance];
    }

    __host__ void copyIn (const Cigars device_cigars) {
        cudaMemcpy(this->data, device_cigars.data, this->cigars_size_bytes(), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERR;
    }

    __host__ void device_reset () {
        cudaMemset(this->data, 0, this->cigars_size_bytes());
        CUDA_CHECK_ERR
    }

    __host__ void print_cigar (const int n) const {
        // Cigar is in reverse order
        edit_cigar_t* curr_cigar = this->get_cigar(n);
        size_t cigar_len = strnlen(curr_cigar, this->max_distance);
        for (int i=(cigar_len - 1); i >= 0; i--) {
            printf("%c", curr_cigar[i]);
        }
        printf("\n");
    }

    __host__ bool check_cigar (const int n, const WF_element element) const {
        const edit_cigar_t* curr_cigar = this->get_cigar(n);
        const size_t cigar_len = strnlen(curr_cigar, this->max_distance);
        int text_pos = 0, pattern_pos = 0;
        // Cigar is reversed
        for (int i=0; i<cigar_len; i++) {
            edit_cigar_t curr_cigar_element = curr_cigar[cigar_len - i - 1];
            switch (curr_cigar_element) {
                case 'M':
                    if (element.pattern[pattern_pos] != element.text[text_pos]) {
                        DEBUG("Alignment not matching at CCIGAR index %d"
                              " (pattern[%d] = %c != text[%d] = %c)",
                              i, pattern_pos, element.pattern[pattern_pos],
                              text_pos, element.text[text_pos]);
                        return false;
                    }
                    ++pattern_pos;
                    ++text_pos;
                    break;
                case 'I':
                    ++text_pos;
                    break;
                case 'D':
                    ++pattern_pos;
                    break;
                case 'X':
                    if (element.pattern[pattern_pos] == element.text[text_pos]) {
                        DEBUG("Alignment not mismatching at CCIGAR index %d"
                              " (pattern[%d] = %c == text[%d] = %c)",
                              i, pattern_pos, element.pattern[pattern_pos],
                              text_pos, element.text[text_pos]);
                        return false;
                    }
                    ++pattern_pos;
                    ++text_pos;
                    break;
                default:
                    WF_FATAL("Invalid CIGAR generated.");
                    break;
            }
        }

        if (pattern_pos != element.len) {
            DEBUG("Alignment incorrect length, pattern-aligned: %d, "
                  "pattern-length: %d.", pattern_pos, element.len);
            return false;
        }

        if (text_pos != element.len) {
            DEBUG("Alignment incorrect length, text-aligned: %d, "
                  "text-length: %d.", text_pos, element.len);
            return false;
        }

        return true;
    }
private:
    size_t cigars_size_bytes () {
        return this->num_cigars * this->max_distance * sizeof(edit_cigar_t);
    }
    size_t max_distance;
    int num_cigars;
    edit_cigar_t *data;
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

    SequenceReader sequences_reader;

    Cigars d_cigars;
    Cigars h_cigars;

    Sequences (WF_element* e, size_t num_e, size_t seq_len, size_t batch_size,
               SequenceReader reader) : \
                                                elements(e),               \
                                                num_elements(num_e),       \
                                                sequence_len(seq_len),     \
    // TODO: Assume error of max 20%, arbitrary chose
                                                max_distance(seq_len / 5), \
                                                batch_size(batch_size),    \
                                                sequences_reader(reader),  \
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
    bool CPU_read_next_sequences ();
    bool GPU_prepare_memory_next_batch ();
    void backtrace ();
    void GPU_launch_wavefront_distance ();

private:
    SEQ_TYPE* sequences_device_ptr;
    ewf_offset_t* offsets_device_ptr;
    ewf_offset_t* offsets_host_ptr;
};

#endif // Header guard WAVEFRONT_H
