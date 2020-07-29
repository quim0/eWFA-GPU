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

#define EWAVEFRONT_V(k,offset) ((offset)-(k))
#define EWAVEFRONT_H(k,offset) (offset)
#define EWAVEFRONT_DIAGONAL(h,v) ((h)-(v))
#define EWAVEFRONT_OFFSET(h,v)   (h)

__host__ __device__ void pprint_wavefront(const edit_wavefronts_t* const wf) {
    const ewf_offset_t* const offsets = wf->offsets;
    const int hi = wf->d;
    const int lo = -wf->d;

    printf("+---+\n");
    int i;
    for (i=lo; i<=hi; i++) {
        printf("|%03d|\n", offsets[i]);
        printf("+---+\n");
    }
}

#if 0
__device__ void WF_backtrace (const edit_wavefronts_t* wavefronts,
                              const Cigars cigars,
                              int target_k) {
    if (threadIdx.x == 0) {
        edit_cigar_t* const curr_cigar = cigars.get_cigar(blockIdx.x);
        int edit_cigar_idx = 0;
        int k = target_k;
        int d = wavefronts->d;
        const ewf_offset_t* curr_offsets_column = wavefronts->offsets;
        ewf_offset_t offset = curr_offsets_column[k];

        while (d > 0) {
            int lo = -(d - 1);
            int hi = (d - 1);
            ewf_offset_t* prev_offsets = OFFSETS_PTR(wavefronts->offsets_base, d - 1);
            if (lo <= k+1 && k+1 <= hi && offset == prev_offsets[k+1]) {
                curr_cigar[edit_cigar_idx++] = 'D';
                k++;
                d--;
            }
            else if (lo <= k-1 && k-1 <= hi && offset == prev_offsets[k-1] + 1) {
                curr_cigar[edit_cigar_idx++] = 'I';
                k--;
                offset--;
                d--;
            }
            else if (lo <= k && k <= hi && offset == prev_offsets[k] + 1) {
                curr_cigar[edit_cigar_idx++] = 'X';
                offset--;
                d--;
            }
            else {
                curr_cigar[edit_cigar_idx++] = 'M';
                offset--;
            }
        }

        while (offset > 0) {
            curr_cigar[edit_cigar_idx++] = 'M';
            offset--;
        }
    }
}
#endif

// Assumes 1 block per element
// TODO: Offsets are stored in an "array of structs" pattern, it'd be more
// efficient for memory acceses to store them as a "struct of arrays" (for loop)
// ^ is this really true ????
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
            const SEQ_TYPE* curr_pattern_ptr = &pattern[v];
            const SEQ_TYPE* curr_text_ptr = &text[h];

            // 0xffffff00
            uintptr_t alignment_mask = (uintptr_t)-1 << 2;
            // Align addresses to 4 bytes
            SEQ_TYPE* curr_pattern_ptr_aligned =
                                (SEQ_TYPE*)((uintptr_t)curr_pattern_ptr & alignment_mask);
            SEQ_TYPE* curr_text_ptr_aligned =
                                (SEQ_TYPE*)((uintptr_t)curr_text_ptr & alignment_mask);

            // 0x3 --> 0b11
            unsigned int pattern_bytes_to_remove = (uintptr_t)curr_pattern_ptr & 0x3;
            unsigned int text_bytes_to_remove = (uintptr_t)curr_text_ptr & 0x3;
            unsigned int bytes_to_remove = max(pattern_bytes_to_remove,
                                               text_bytes_to_remove);

            uint32_t packed_pattern = *(uint32_t*)curr_pattern_ptr_aligned;
            uint32_t packed_text = *(uint32_t*)curr_text_ptr_aligned;

            // Rightshift because the data is little-endian encoded
            packed_pattern = packed_pattern >> (pattern_bytes_to_remove * 8);
            packed_text = packed_text >> (text_bytes_to_remove * 8);

            uint32_t xored_val = packed_pattern ^ packed_text;
            uint32_t b0 = (xored_val & 0xff) << 24;
            uint32_t b1 = (xored_val & 0xff00) << 8;
            uint32_t b2 = (xored_val & 0xff0000) >> 8;
            uint32_t b3 = (xored_val & 0xff000000) >> 24;
            xored_val  = b0 | b1 | b2 | b3;

            int lz = __clz(xored_val);
            int useful_bytes = 4 - bytes_to_remove;
            acc += min(lz / 8, useful_bytes);
            if (lz < (useful_bytes * 8)) {
                goto end_extend;
            }

            v += useful_bytes;
            h += useful_bytes;
        }

        while (v < plen && h < tlen && pattern[v++] == text[h++])
            acc++;

end_extend:
        offsets[k] += acc;
    }
    __syncthreads();
}

__device__ void WF_compute_kernel (edit_wavefronts_t* const wavefronts,
                                   edit_wavefronts_t* const next_wavefronts,
                                   WF_backtrace_t* const backtraces) {
    ewf_offset_t * const offsets = wavefronts->offsets;
    ewf_offset_t * const next_offsets = next_wavefronts->offsets;

    int tid = threadIdx.x;
    int tnum = blockDim.x;

    const int hi = wavefronts->d;
    const int lo = -wavefronts->d;

    __syncthreads();

    for(int k = lo + tid - 1; k <= hi + 1; k += tnum) {
        const ewf_offset_t max_ins_sub = max(offsets[k], offsets[k - 1]) + 1;
        next_offsets[k] = max(max_ins_sub, offsets[k + 1]);

        // Calculate the partial backtraces
        ewf_offset_t del = next_offsets[k] ^ offsets[k + 1];
        ewf_offset_t sub = next_offsets[k] ^ offsets[k];
        ewf_offset_t ins = next_offsets[k] ^ offsets[k - 1];

        del = del * (del == 0);
        sub = sub * (sub == 0);
        ins = ins * (ins == 0);

        int curr_d = wavefronts->d + 1;
        WRITE_BT_OP(backtraces, curr_d, del | sub | ins);
    }

    if (tid == 0)
        next_wavefronts->d = wavefronts->d + 1;
    __syncthreads();
}

__global__ void WF_edit_distance (const WF_element* elements,
                                  SEQ_TYPE* sequences_base_ptr,
                                  const size_t max_distance,
                                  const size_t max_seq_len,
                                  WF_backtrace_t* backtraces_base_ptr,
                                  WF_backtrace_t* result_backtraces) {
    extern __shared__ uint8_t shared_mem_chunk[];
    const int tid = threadIdx.x;
    const WF_element element = elements[blockIdx.x];
    WF_backtrace_t* const backtraces = &backtraces_base_ptr[blockIdx.x];

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

    ewf_offset_t* const shared_offsets =
                    (ewf_offset_t*)(&shared_mem_chunk[max_seq_len * 2]);
    ewf_offset_t* const shared_next_offsets = 
                    (ewf_offset_t*)(&shared_offsets[WF_ELEMENTS(max_distance)]);
    // Initialize the offsets to -1 to avoid loop peeling on compute
    for (int i=tid; i<WF_ELEMENTS(max_distance); i += blockDim.x) {
        shared_offsets[i] = -1;
        shared_next_offsets[i] = -1;
    }

    // + max_distance to center the offsets
    edit_wavefronts_t wavefronts_obj =
                 {.d = 0, .offsets = shared_offsets + max_distance};
    edit_wavefronts_t next_wavefronts_obj =
                 {.d = 0, .offsets = shared_next_offsets + max_distance};

    // Use pointers to be able to swap them efficiently
    edit_wavefronts_t* wavefronts = &wavefronts_obj;
    edit_wavefronts_t* next_wavefronts = &next_wavefronts_obj;

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
        if (target_k_abs <= distance && wavefronts->offsets[target_k] == target_offset)
            break;

        WF_compute_kernel(wavefronts, next_wavefronts, backtraces);

        // SWAP
        edit_wavefronts_t* tmp = next_wavefronts;
        next_wavefronts = wavefronts;
        wavefronts = tmp;
    }
    //WF_backtrace(wavefronts, cigars, target_k);

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
