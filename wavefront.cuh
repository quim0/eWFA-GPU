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

#define EWAVEFRONT_V(k,offset) ((offset)-(k))
#define EWAVEFRONT_H(k,offset) (offset)
#define EWAVEFRONT_DIAGONAL(h,v) ((h)-(v))
#define EWAVEFRONT_OFFSET(h,v)   (h)

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
                  this->cigars_results_size() / (1 << 20));
            cudaMalloc((void**) &(this->data), this->cigars_results_size());
            CUDA_CHECK_ERR
        }
        else {
            cudaMallocHost((void**)&this->data, this->cigars_results_size());
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

    __host__ __device__ WF_backtrace_result_t* get_backtraces (const int n) const {
        // Get the backtraces of alignment "n"
        return &this->data[n];
    }

    __host__ edit_cigar_t* get_cigar_ascii (const int n) const {
        return &this->cigars_ascii[n * this->cigar_max_len];
    }

    __host__ void copyIn (const Cigars device_cigars) {
        cudaMemcpy(this->data, device_cigars.data, this->cigars_results_size(), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERR;
        DEBUG("Copied the packed backtraces from the device to host.");
    }

    __host__ void device_reset () {
        cudaMemset(this->data, 0, this->cigars_results_size());
        CUDA_CHECK_ERR
    }

    __host__ void reset () {
        memset(this->cigars_ascii, 0, this->cigars_size_bytes());
    }

#if 0
    __host__ void print_cigar (const int n) const {
        // Cigar is in reverse order
        edit_cigar_t* curr_cigar = this->get_cigar_ascii(n);
        size_t cigar_len = strnlen(curr_cigar, this->cigar_max_len);
        for (int i=(cigar_len - 1); i >= 0; i--) {
            printf("%c", curr_cigar[i]);
        }
        printf("\n");
    }
#endif

	__host__ ewf_offset_t extend_wavefront (
            const ewf_offset_t offset_val,
            const int curr_k,
            const char* const pattern,
            const int pattern_length,
            const char* const text,
            const int text_length) {
        // Parameters
        int v = EWAVEFRONT_V(curr_k, offset_val);
        int h = EWAVEFRONT_H(curr_k, offset_val);
        ewf_offset_t acc = 0;
        while (v<pattern_length && h<text_length && pattern[v++]==text[h++]) {
          acc++;
        }
        return acc;
	}

    __host__ edit_cigar_t* generate_ascii_cigar (const int n,
                                        const SEQ_TYPE* text,
                                        const int text_length,
                                        const SEQ_TYPE* pattern,
                                        const int pattern_length) {
        // Generate the ASCII cigar from the packed one, get the "n"th cigar

        WF_backtrace_result_t* backtrace_packed = this->get_backtraces(n);
        const size_t distance = backtrace_packed->distance;

        edit_cigar_t* cigar_ascii = this->get_cigar_ascii(n);

        // Assume that the packed cigar have two 64 bit words
		ewf_offset_t curr_off_value = 0;
        int curr_k = 0;
        for (int d=0; d<distance; d++) {
            // Compute the extend
            ewf_offset_t acc = this->extend_wavefront(curr_off_value,
                                                     curr_k,
                                                     pattern,
                                                     pattern_length,
                                                     text,
                                                     text_length);
            for (int j=0; j<acc; j++) {
                *cigar_ascii = 'M';
                cigar_ascii++;
            }

            curr_off_value += acc;

            const int w_idx = d / 32;
            const int w_mod = d % 32;

            // 3 --> 0b11
            uint64_t op = (uint64_t)
                            ((backtrace_packed->words[w_idx] >> (w_mod * 2)) & 3);
            switch (op) {
                // k + 1
                case OP_DEL:
                    *cigar_ascii = 'D';
                    curr_k--;
                    break;
                // k
                case OP_SUB:
                    *cigar_ascii = 'X';
                    curr_off_value++;
                    break;
                // k  - 1
                case OP_INS:
                    *cigar_ascii = 'I';
                    curr_k++;
                    curr_off_value++;
                    break;
                default:
                    DEBUG("Incorrect operation found: %lu", op);
                    WF_FATAL("Invalid packed value in backtrace\n");
            }
            cigar_ascii++;
        }

        // Make one last extension
        ewf_offset_t acc = this->extend_wavefront(curr_off_value,
                                                 curr_k,
                                                 pattern,
                                                 pattern_length,
                                                 text,
                                                 text_length);
        for (int j=0; j<acc; j++) {
            *cigar_ascii = 'M';
            cigar_ascii++;
        }

        return this->get_cigar_ascii(n);
    }

    __host__ bool check_cigar (const int n, const WF_element element,
                               const SEQ_TYPE* seq_base_ptr,
                               const size_t max_seq_len) {
        int text_pos = 0, pattern_pos = 0;
        SEQ_TYPE* text = TEXT_PTR(element.alignment_idx, seq_base_ptr,
                                  max_seq_len);
        SEQ_TYPE* pattern = PATTERN_PTR(element.alignment_idx, seq_base_ptr,
                                        max_seq_len);

        const edit_cigar_t* curr_cigar = this->generate_ascii_cigar(n,
                                                        text,
                                                        element.tlen,
                                                        pattern,
                                                        element.plen);
        if (!curr_cigar)
            return false;
#ifdef DEBUG_MODE
        if (n == 0) {
            DEBUG("CIGAR: %s", curr_cigar);
        }
#endif
        const size_t cigar_len = strnlen(curr_cigar, this->cigar_max_len);

        for (int i=0; i<cigar_len; i++) {
            edit_cigar_t curr_cigar_element = curr_cigar[i];
            switch (curr_cigar_element) {
                case 'M':
                    if (pattern[pattern_pos] != text[text_pos]) {
                        DEBUG("Alignment not matching at CCIGAR index %d"
                              " (pattern[%d] = %c != text[%d] = %c)",
                              i, pattern_pos, pattern[pattern_pos],
                              text_pos, text[text_pos]);
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
                        DEBUG("Alignment not mismatching at CCIGAR index %d"
                              " (pattern[%d] = %c == text[%d] = %c)",
                              i, pattern_pos, pattern[pattern_pos],
                              text_pos, text[text_pos]);
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
            DEBUG("Alignment incorrect length, pattern-aligned: %d, "
                  "pattern-length: %zu. (n: %d)", pattern_pos, element.plen, n);
            DEBUG("TEXT: %s", text);
            DEBUG("PATTERN: %s", pattern);
            return false;
        }

        if (text_pos != element.tlen) {
            DEBUG("Alignment incorrect length, text-aligned: %d, "
                  "text-length: %zu", text_pos, element.tlen);
            return false;
        }

        return true;
    }
private:
    size_t cigars_size_bytes () {
        return this->num_cigars * this->cigar_max_len * sizeof(edit_cigar_t);
    }
    size_t cigars_results_size () {
        return this->num_cigars * sizeof(WF_backtrace_result_t);
    }
    size_t max_distance;
    size_t cigar_max_len;
    int num_cigars;
    WF_backtrace_result_t* data;
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
