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
#include <inttypes.h>

// Assumes 1 block per element
__device__ ewf_offset_t WF_extend_kernel (const WF_element element,
                                  const SEQ_TYPE* text,
                                  const SEQ_TYPE* pattern,
                                  const int tlen,
                                  const int plen,
                                  const int k,
                                  const ewf_offset_t offset_k) {
    int v  = EWAVEFRONT_V(k, offset_k);
    int h  = EWAVEFRONT_H(k, offset_k);

    const int bases_to_cmp = 16;
    int eq_elements = 0;
    int acc = 0;
    // Compare 16 bases at once
    while (v < plen && h < tlen) {
        // Which byte to pick
        int real_v = v / 4;
        int real_h = h / 4;

        // Get the displacement inside the aligned word
        int pattern_displacement = v % bases_to_cmp;
        int text_displacement = h % bases_to_cmp;

        // 0xffffffffffffff00
        uintptr_t alignment_mask = (uintptr_t)-1 << 2;
        uint32_t* word_p_ptr = (uint32_t*)((uintptr_t)(pattern + real_v) & alignment_mask);
        uint32_t* next_word_p_ptr = word_p_ptr + 1;
        uint32_t* word_t_ptr = (uint32_t*)((uintptr_t)(text + real_h) & alignment_mask);
        uint32_t* next_word_t_ptr = word_t_ptr + 1;


        // * 2 because each element is 2 bits
        uint32_t sub_word_p_1 = *word_p_ptr;
        uint32_t sub_word_p_2 = *next_word_p_ptr;
        sub_word_p_1 = sub_word_p_1 << (pattern_displacement * 2);
        // Convert the u32 to big-endian, as little endian inverts the order
        // for the sequences.
        sub_word_p_2 = *next_word_p_ptr;
        // Cast to uint64_t is done to avoid undefined behaviour in case
        // it's shifted by 32 elements.
        // ----
        // The type of the result is that of the promoted left operand. The
        // behavior is undefined if the right operand is negative, or
        // greater than or equal to the length in bits of the promoted left
        // operand.
        // ----
        sub_word_p_2 = ((uint64_t)sub_word_p_2) >>
            ((bases_to_cmp - pattern_displacement) * 2);

        uint32_t sub_word_t_1 = *word_t_ptr;
        sub_word_t_1 = sub_word_t_1 << (text_displacement * 2);
        uint32_t sub_word_t_2 = *next_word_t_ptr;
        sub_word_t_2 = ((uint64_t)sub_word_t_2) >>
            ((bases_to_cmp - text_displacement) * 2);

        uint32_t word_p = sub_word_p_1 | sub_word_p_2;
        uint32_t word_t = sub_word_t_1 | sub_word_t_2;

        uint32_t diff = word_p ^ word_t;
        // Branchless method to remove the equal bits if we read "too far away"
        // TODO: Find a better way to implement this?
        uint32_t full_mask = (uint32_t)-1;
        int next_v = v + bases_to_cmp;
        int next_h = h + bases_to_cmp;
        uint32_t mask_p = full_mask << ((next_v - plen) * 2 * (next_v > plen));
        uint32_t mask_t = full_mask << ((next_h - tlen) * 2 * (next_h > tlen));
        diff = diff | ~mask_p | ~mask_t;

        int lz = __clz(diff);

        // each element has 2 bits
        eq_elements = lz / 2;
        acc += eq_elements;

        if (eq_elements < bases_to_cmp) {
            break;
        }

        v += bases_to_cmp;
        h += bases_to_cmp;
    }

    return offset_k + acc;
}

__device__ void WF_compute_kernel (ewf_offset_t* const offsets,
                                   ewf_offset_t* const next_offsets,
                                   WF_backtrace_t* const backtraces,
                                   WF_backtrace_t* const next_backtraces,
                                   const WF_element element,
                                   const SEQ_TYPE* text,
                                   const SEQ_TYPE* pattern,
                                   const int tlen,
                                   const int plen,
                                   const int distance) {
    int tid = threadIdx.x;
    int tnum = blockDim.x;

    const int hi = distance;
    const int lo = -distance;

    for(int k = lo + tid; k <= hi; k += tnum) {
        // Load the three necesary values on registers
        const ewf_offset_t off_k = offsets[k];
        const ewf_offset_t off_k_up = offsets[k + 1];
        const ewf_offset_t off_k_down = offsets[k - 1];

        const ewf_offset_t max_ins_sub = max(off_k, off_k_down) + 1;
        const ewf_offset_t next_off_k = max(max_ins_sub, off_k_up);

        // Calculate the partial backtraces
        ewf_offset_t del = off_k_up;
        ewf_offset_t sub = off_k + 1;
        ewf_offset_t ins = off_k_down + 1;

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
        // -1 because operation 0 is at distance 1
        int curr_d = distance - 1;
        int word = curr_d / 32;
        curr_bt->words[0] = prev_bt->words[0];
        curr_bt->words[1] = prev_bt->words[1];
        curr_bt->words[word] |= (((uint64_t)op) <<  ((curr_d % 32) * 2));

        const ewf_offset_t res = WF_extend_kernel(
                 element, text, pattern,
                 tlen, plen, k, next_off_k);
        next_offsets[k] = res;
    }

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

    SEQ_TYPE* const shared_pattern = (SEQ_TYPE*)&shared_mem_chunk[0];
    SEQ_TYPE* const shared_text = (SEQ_TYPE*)&shared_mem_chunk[max_seq_len];

    int text_len = element.tlen;
    int pattern_len = element.plen;
    SEQ_TYPE* text = TEXT_PTR(blockIdx.x, sequences_base_ptr, max_seq_len);
    SEQ_TYPE* pattern = PATTERN_PTR(blockIdx.x, sequences_base_ptr, max_seq_len);

    // Pattern and text are contiguous in memory
    for (int i=tid*4; i<max_seq_len*2; i += blockDim.x*4) {
        *(uint32_t*)(&shared_pattern[i]) = *(uint32_t*)(&pattern[i]);
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
    ewf_offset_t* wf_offsets = shared_offsets + max_distance + 1;
    ewf_offset_t* next_wf_offsets = shared_next_offsets + max_distance + 1;

    // TODO: Remove magic number
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

    const int target_k = EWAVEFRONT_DIAGONAL(text_len, pattern_len);
    const int target_k_abs = (target_k >= 0) ? target_k : -target_k;
    const ewf_offset_t target_offset = EWAVEFRONT_OFFSET(text_len, pattern_len);

    int distance = 0;

    if (tid == 0) {
        ewf_offset_t res = WF_extend_kernel(element, text, pattern,
                                            text_len, pattern_len, 0, 0);
        wf_offsets[0] = res;
    }

    for (distance = 1; distance < max_distance; distance++) {
        // Computes does compute + extend per diagonal
        // TODO: Change function name
        WF_compute_kernel(wf_offsets, next_wf_offsets, backtraces,
                  next_backtraces, element, shared_text, shared_pattern,
                  text_len, pattern_len, distance);

        // Compare against next_offsets becuse the extend updates that
        if (target_k_abs <= distance && next_wf_offsets[target_k] == target_offset) {
            break;
        }

        // SWAP offsets
        ewf_offset_t* offset_tmp = next_wf_offsets;
        next_wf_offsets = wf_offsets;
        wf_offsets = offset_tmp;

        // SWAP backtraces
        WF_backtrace_t* bt_tmp = next_backtraces;
        next_backtraces = backtraces;
        backtraces = bt_tmp;
    }

    WF_backtrace_result_t *curr_res = cigars.get_backtraces(blockIdx.x);
    curr_res->distance = distance;
    // TODO: Check if it's in next_backtraces for all cases. It should be
    curr_res->words[0] = next_backtraces[target_k].words[0];
    curr_res->words[1] = next_backtraces[target_k].words[1];

#ifdef DEBUG_MODE
    if (blockIdx.x == 0 && tid == 0) {
        if (distance == max_distance) {
            // TODO
            printf("Max distance reached!!!\n");
        }
        printf("Distance: %d\n", distance);
    }
#endif
}

// threadIdx.y decides if the thread works on the pattern (0) or text (1)
__global__ void compact_sequences(const SEQ_TYPE* const sequences_in,
                                  SEQ_TYPE* const sequences_out,
                                  const size_t max_seq_len_unpacked,
                                  const size_t max_seq_len_packed) {
    const int alignment_idx = blockIdx.x;

    // Get pointers of the ASCII sequences
    const SEQ_TYPE* pattern_unpacked = PATTERN_PTR(alignment_idx, sequences_in,
                                                   max_seq_len_unpacked);

    // Get pointers of the packed sequences, to be computed
    uint8_t* const pattern_packed = (uint8_t*)PATTERN_PTR(alignment_idx,
                                                   sequences_out,
                                                   max_seq_len_packed);
    uint8_t* sequence_packed = pattern_packed;
    if (threadIdx.y == 1) {
        // use text instead of pattern
        sequence_packed += max_seq_len_packed;
    }

    // Size of pattern+text
    extern __shared__ uint8_t sequences_sh[];

    // Linearize thread ids to copy the text and pattern
    int cpytid = threadIdx.y*blockDim.x + threadIdx.x;
    for (int i=cpytid; i<(max_seq_len_packed*2); i += blockDim.x*blockDim.y) {
        *(uint32_t*)(&sequences_sh[i*4]) = *(uint32_t*)(&pattern_unpacked[i*4]);
    }

    uint8_t* sequence_unpacked = sequences_sh;
    if (threadIdx.y == 1) {
        // use text instead of pattern
        sequence_unpacked += max_seq_len_unpacked;
    }


    // Each thread packs 4 bytes into 1 byte.
    for (int i=threadIdx.x; i<max_seq_len_packed; i += blockDim.x) {
        uint32_t bases = *((uint32_t*)(sequence_unpacked + i*4));
        if (bases == 0)
            break;

        // Extract bases SIMD like --> (base & 6) >> 1 for each element
        bases = (bases & 0x06060606) >> 1;

        const uint8_t base0 = bases & 0xff;
        const uint8_t base1 = (bases >> 8) & 0xff;
        const uint8_t base2 = (bases >> 16) & 0xff;
        const uint8_t base3 = (bases >> 24) & 0xff;

        // Reverse the bases, because they are read in little endian
        uint8_t packed_reg = base3;
        packed_reg |= (base2 << 2);
        packed_reg |= (base1 << 4);
        packed_reg |= (base0 << 6);

        // Byte idx if we were packing the sequences in big endian
        const int be_idx = i;

        // Save to the correct by to be little endian encoded for 32bits ints
        int le_byte_idx;
        switch (be_idx % 4) {
            case 0:
                le_byte_idx = be_idx + (3 - (be_idx % 4));
                break;
            case 1:
                le_byte_idx = be_idx + (be_idx % 4);
                break;
            case 2:
                le_byte_idx = be_idx - (3 - (be_idx % 4));
                break;
            case 3:
            default:
                le_byte_idx = be_idx - (be_idx % 4);
                break;
        }

        sequence_packed[le_byte_idx] = packed_reg;
    }
}
