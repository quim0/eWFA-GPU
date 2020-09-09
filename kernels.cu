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

#include "wavefront.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

// Assumes 1 block per element
__device__ void WF_extend_kernel (const WF_element element,
                                  edit_wavefronts_t* const wavefronts,
                                  SEQ_TYPE* shared_text,
                                  SEQ_TYPE* shared_pattern,
                                  int tlen,
                                  int plen) {
    ewf_offset_t * const offsets = wavefronts->offsets;

    int tid = threadIdx.x;
    int tnum = blockDim.x;

    const SEQ_TYPE* text = shared_text;
    const SEQ_TYPE* pattern = shared_pattern;

    const int k_min = -wavefronts->d;
    const int k_max = wavefronts->d;

    // Access the different "k" values in a coescaled way.
    // This will only loop in the case the number of threads assigned to this
    // block is lower than the current number of "k" being processed.
    for(int k = k_min + tid; k <= k_max; k += tnum) {
        int v  = EWAVEFRONT_V(k, offsets[k]);
        int h  = EWAVEFRONT_H(k, offsets[k]);

        int acc = 0;
        while ((v + 4) < plen && (h + 4) < tlen) {
            int real_v = v / 4;
            int real_h = h / 4;
            // v % 4
            int pattern_displacement = v & 3;
            int text_displacement = h & 3;
            int displacement_elements  = max(pattern_displacement, text_displacement);

            // * 2 because each element is 2 bits
            uint8_t byte_p = pattern[real_v] << (pattern_displacement * 2);
            uint8_t byte_t = text[real_h] << (text_displacement * 2);

            uint8_t diff = byte_p ^ byte_t;
            // diff gets converted to int (32 bits), so it's necessary to remove
            // the first 3 bytes of zeroes
            int lz = __clz(diff) - 24;
            int useful_elements = 4 - displacement_elements;
            // each element has 2 bits
            int eq_elements = min(lz / 2, useful_elements);
            acc += eq_elements;

            if (eq_elements < useful_elements) {
                goto end_extend;
            }

            v += eq_elements;
            h += eq_elements;
        }

        // TODO: Don't do this one by one
        // Get the last elements in case is not a multiple of 4
        while (v < plen && h < tlen) {
            int real_v = v / 4;
            int real_h = h / 4;
            // v % 4
            int pattern_displacement = (3 - (v & 3)) * 2;
            int text_displacement = (3 - (h & 3)) * 2;

            // Get only one position
            uint8_t bits_p = (pattern[real_v] >> pattern_displacement) & 3;
            uint8_t bits_t = (text[real_h] >> text_displacement) & 3;

            if (bits_p == bits_t)
                acc++;
            else
                goto end_extend;

            v++;
            h++;
        }

end_extend:

        offsets[k] += acc;
    }
    __syncthreads();
}

__device__ void WF_compute_kernel (edit_wavefronts_t* const wavefronts,
                                   edit_wavefronts_t* const next_wavefronts,
                                   WF_backtrace_t* const backtraces,
                                   WF_backtrace_t* const next_backtraces) {
    ewf_offset_t * const offsets = wavefronts->offsets;
    ewf_offset_t * const next_offsets = next_wavefronts->offsets;

    int tid = threadIdx.x;
    int tnum = blockDim.x;

    const int hi = wavefronts->d;
    const int lo = -wavefronts->d;

    for(int k = lo + tid - 1; k <= hi + 1; k += tnum) {
        const ewf_offset_t max_ins_sub = max(offsets[k], offsets[k - 1]) + 1;
        next_offsets[k] = max(max_ins_sub, offsets[k + 1]);

        // Calculate the partial backtraces
        ewf_offset_t del = offsets[k + 1];
        ewf_offset_t sub = offsets[k] + 1;
        ewf_offset_t ins = offsets[k - 1] + 1;

        // Set piggy-back identifier
        ewf_offset_t del_p = (del << 2) | OP_DEL; // OP_DEL = 3
        ewf_offset_t sub_p = (sub << 2) | OP_SUB; // OP_SUB = 2
        ewf_offset_t ins_p = (ins << 2) | OP_INS; // OP_INS = 1

        ewf_offset_t tmp_max = max(del_p, sub_p);
        ewf_offset_t max_p = max(tmp_max, ins_p);

        // Recover where the max comes from, 3 --> 0b11
        WF_backtrace_op_t op = (WF_backtrace_op_t)(max_p & 3);

        WF_backtrace_t* curr_bt = next_backtraces + k;
        WF_backtrace_t* prev_bt = &backtraces[k + (op - 2)];

        // set in next_backtraces the correct operation
        int curr_d = wavefronts->d;
        int word = curr_d / 32;
        curr_bt->words[0] = prev_bt->words[0];
        curr_bt->words[1] = prev_bt->words[1];
        curr_bt->words[word] |= (((uint64_t)op) <<  ((curr_d % 32) * 2));
    }

    if (tid == 0)
        next_wavefronts->d = wavefronts->d + 1;
    __syncthreads();
}

__global__ void WF_edit_distance (const WF_element* elements,
                                  SEQ_TYPE* sequences_base_ptr,
                                  const size_t max_distance,
                                  const size_t max_seq_len,
                                  Cigars cigars) {
    extern __shared__ uint8_t shared_mem_chunk[];
    const int tid = threadIdx.x;
    const WF_element element = elements[blockIdx.x];

    SEQ_TYPE* const shared_text = (SEQ_TYPE*)&shared_mem_chunk[0];
    SEQ_TYPE* const shared_pattern = (SEQ_TYPE*)&shared_mem_chunk[max_seq_len];

    int text_len = element.tlen;
    int pattern_len = element.plen;
    SEQ_TYPE* text = TEXT_PTR(blockIdx.x, sequences_base_ptr, max_seq_len);
    SEQ_TYPE* pattern = PATTERN_PTR(blockIdx.x, sequences_base_ptr, max_seq_len);

    // Put text and pattern to shared memory
    for (int i=tid; i<max_seq_len; i += blockDim.x) {
        shared_text[i] = text[i];
        shared_pattern[i] = pattern[i];
    }

    // Skip text/pattern
    ewf_offset_t* const shared_offsets =
                    (ewf_offset_t*)(&shared_mem_chunk[max_seq_len * 2]);
    // +2 to avoid loop peeling in the compute function
    ewf_offset_t* const shared_next_offsets = 
                    (ewf_offset_t*)(&shared_offsets[WF_ELEMENTS(max_distance) + 2]);
    // Initialize the offsets to -1 to avoid loop peeling on compute
    for (int i=tid; i<WF_ELEMENTS(max_distance) + 2; i += blockDim.x) {
        shared_offsets[i] = -1;
        shared_next_offsets[i] = -1;
    }

    // + max_distance to center the offsets, +1 to avoid loop peeling in the
    // compute function
    edit_wavefronts_t wavefronts_obj =
                 {.d = 0, .offsets = shared_offsets + max_distance + 1};
    edit_wavefronts_t next_wavefronts_obj =
                 {.d = 0, .offsets = shared_next_offsets + max_distance + 1};

    // Use pointers to be able to swap them efficiently
    edit_wavefronts_t* wavefronts = &wavefronts_obj;
    edit_wavefronts_t* next_wavefronts = &next_wavefronts_obj;

    // Get backtraces from shared memory, skip the last offset

    // TODO: Why dynamically allocated shared memory launch unaligned address
    // error?
    __shared__ WF_backtrace_t backtraces_shared[WF_ELEMENTS(64)];
    __shared__ WF_backtrace_t next_backtraces_shared[WF_ELEMENTS(64)];
    for (int i=tid; i<WF_ELEMENTS(64); i += blockDim.x) {
        backtraces_shared[i] = {0};
        next_backtraces_shared[i] = {0};
    }

    // We can not swap arrays so declare pointers here, also it's needed for
    // centering the array
    WF_backtrace_t* backtraces = backtraces_shared + max_distance;
    WF_backtrace_t* next_backtraces = next_backtraces_shared + max_distance;

    // The first offsets must be 0, not -1
    if (tid == 0)
        wavefronts->offsets[0] = 0;

    __syncthreads();

    const int target_k = EWAVEFRONT_DIAGONAL(text_len, pattern_len);
    const int target_k_abs = (target_k >= 0) ? target_k : -target_k;
    const ewf_offset_t target_offset = EWAVEFRONT_OFFSET(text_len, pattern_len);
    int distance;

    for (distance = 0; distance < max_distance; distance++) {
        wavefronts->d = distance;
        WF_extend_kernel(element, wavefronts, shared_text, shared_pattern,
                         text_len, pattern_len);
        if (target_k_abs <= distance && wavefronts->offsets[target_k] == target_offset) {
            break;
        }

        WF_compute_kernel(wavefronts, next_wavefronts, backtraces, next_backtraces);

        // SWAP offsets
        edit_wavefronts_t* offset_tmp = next_wavefronts;
        next_wavefronts = wavefronts;
        wavefronts = offset_tmp;

        // SWAP backtraces
        WF_backtrace_t* bt_tmp = next_backtraces;
        next_backtraces = backtraces;
        backtraces = bt_tmp;
    }

    WF_backtrace_result_t *curr_res = cigars.get_backtraces(blockIdx.x);
    curr_res->distance = distance;
    curr_res->words[0] = backtraces[target_k].words[0];
    curr_res->words[1] = backtraces[target_k].words[1];

#ifdef DEBUG_MODE
    if (blockIdx.x == 0 && tid == 0) {
        if (distance == max_distance) {
            // TODO
            printf("Max distance reached!!!\n");
        }
        char tmp = text[text_len];
        text[text_len] = '\0';
        printf("TEXT: %s\n", text);
        text[text_len] = tmp;
        tmp = pattern[pattern_len];
        pattern[pattern_len] = '\0';
        printf("PATTERN: %s\n", pattern);
        pattern[pattern_len] = tmp;
        printf("Distance: %d\n", distance);
    }
#endif
}
