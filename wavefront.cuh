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
    __host__ Cigars (int n, size_t max_d, size_t cigar_max_len, bool device) {
        this->max_distance = max_d;
        this->num_cigars = n;
        // +1 to store a nullbyte
        this->cigar_max_len = cigar_max_len + 1;
        if (device) {
            DEBUG("Allocating %zu MiB to store the partial backtraces",
                  this->cigars_size_packed() / (1 << 20));
            cudaMalloc((void**) &(this->data), this->cigars_size_packed());
            CUDA_CHECK_ERR
            cudaMemset(this->data, 0, this->cigars_size_packed());
            CUDA_CHECK_ERR
        }
        else {
            cudaMallocHost((void**)&this->data, this->cigars_size_packed());
            if (!this->data) {
                fprintf(stderr, "Out of memory on host. (%s:%d)", __FILE__, __LINE__);
                exit(1);
            }
            this->cigars_ascii = (edit_cigar_t*)calloc(this->cigars_size_bytes(), 1);
            if (!this->cigars_ascii) {
                fprintf(stderr, "Out of memory on host. (%s:%d)", __FILE__, __LINE__);
                exit(1);
            }
        }
    }

    __host__ __device__ WF_backtrace_t* get_cigar_packed (const int n) const {
        return &this->data[n];
    }

    __host__ edit_cigar_t* get_cigar_ascii (const int n) const {
        return &this->cigars_ascii[n * this->cigar_max_len];
    }

    __host__ void copyIn (const Cigars device_cigars) {
        cudaMemcpy(this->data, device_cigars.data, this->cigars_size_packed(), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERR;
    }

    __host__ void device_reset () {
        cudaMemset(this->data, 0, this->cigars_size_packed());
        CUDA_CHECK_ERR
    }

    __host__ void print_cigar (const int n) const {
        // Cigar is in reverse order
        edit_cigar_t* curr_cigar = this->get_cigar_ascii(n);
        size_t cigar_len = strnlen(curr_cigar, this->cigar_max_len);
        for (int i=(cigar_len - 1); i >= 0; i--) {
            printf("%c", curr_cigar[i]);
        }
        printf("\n");
    }

    __host__ bool check_cigar (const int n, const WF_element element,
                               const SEQ_TYPE* seq_base_ptr,
                               const size_t max_seq_len) const {
        const edit_cigar_t* curr_cigar = this->get_cigar_ascii(n);
        const size_t cigar_len = strnlen(curr_cigar, this->cigar_max_len);
        int text_pos = 0, pattern_pos = 0;
        SEQ_TYPE* text = TEXT_PTR(element.alignment_idx, seq_base_ptr,
                                  max_seq_len);
        SEQ_TYPE* pattern = PATTERN_PTR(element.alignment_idx, seq_base_ptr,
                                        max_seq_len);
        // Cigar is reversed
        for (int i=0; i<cigar_len; i++) {
            edit_cigar_t curr_cigar_element = curr_cigar[cigar_len - i - 1];
            switch (curr_cigar_element) {
                case 'M':
                    if (pattern[pattern_pos] != text[text_pos]) {
                        //DEBUG("Alignment not matching at CCIGAR index %d"
                        //      " (pattern[%d] = %c != text[%d] = %c)",
                        //      i, pattern_pos, pattern[pattern_pos],
                        //      text_pos, text[text_pos]);
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
                    if (pattern[pattern_pos] == text[text_pos]) {
                        //DEBUG("Alignment not mismatching at CCIGAR index %d"
                        //      " (pattern[%d] = %c == text[%d] = %c)",
                        //      i, pattern_pos, pattern[pattern_pos],
                        //      text_pos, text[text_pos]);
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

        if (pattern_pos != element.plen) {
            //DEBUG("Alignment incorrect length, pattern-aligned: %d, "
            //      "pattern-length: %zu.", pattern_pos, element.plen);
            return false;
        }

        if (text_pos != element.tlen) {
            //DEBUG("Alignment incorrect length, text-aligned: %d, "
            //      "text-length: %zu", text_pos, element.tlen);
            return false;
        }

        return true;
    }
private:
    size_t cigars_size_bytes () {
        return this->num_cigars * this->cigar_max_len * sizeof(edit_cigar_t);
    }
    size_t cigars_size_packed () {
        // *2 to have backtraces and next_backtraces per alignment, same
        // strategy as with offsets (having two pointers swapping)
        return 2 * this->num_cigars * WF_ELEMENTS(this->max_distance) * sizeof(WF_backtrace_t);
    }
    size_t max_distance;
    size_t cigar_max_len;
    int num_cigars;
    WF_backtrace_t* data;
    edit_cigar_t* cigars_ascii;
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
    size_t batch_size;
    // Current batch index
    int batch_idx;
    // Where to start to read the alignments
    int initial_alignment;

    SequenceReader sequences_reader;

    Cigars d_cigars;
    Cigars h_cigars;

    cudaStream_t HtD_stream;

    Sequences (WF_element* e, size_t num_e, size_t seq_len, size_t batch_size,
               SequenceReader reader, int initial_alignment) :         \
                                            elements(e),               \
                                            num_elements(num_e),       \
                                            // Max distance 64 because, right
                                            // now, we have 128 bits of partial
                                            // backtrace per diagonal
                                            max_distance(64), \
                                            batch_size(batch_size),    \
                                            sequences_reader(reader),  \
                                            initial_alignment(initial_alignment), \
                                            batch_idx(0),              \
                                            d_cigars(
                                                Cigars(batch_size,
                                                       max_distance,
                                                       seq_len * 2,
                                                       true)),         \
                                            h_cigars(
                                                Cigars(batch_size,
                                                       max_distance,
                                                       seq_len * 2,
                                                       false)) {
        // TODO: Destroy stream
        cudaStreamCreate(&this->HtD_stream);
        CUDA_CHECK_ERR
    }
    bool GPU_memory_init ();
    bool GPU_memory_free ();
    void backtrace ();
    void GPU_launch_wavefront_distance ();
    bool prepare_next_batch ();

private:
    bool CPU_read_next_sequences ();
    bool GPU_prepare_memory_next_batch ();
    SEQ_TYPE* sequences_device_ptr;
};

#endif // Header guard WAVEFRONT_H
