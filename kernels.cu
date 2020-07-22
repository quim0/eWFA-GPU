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
    const ewf_offset_t* const offsets = OFFSETS_PTR(wf->offsets_base, wf->d);
    const int hi = wf->d;
    const int lo = -wf->d;

    printf("+---+\n");
    int i;
    for (i=lo; i<=hi; i++) {
        printf("|%03d|\n", offsets[i]);
        printf("+---+\n");
    }
}

__device__ void WF_backtrace (const edit_wavefronts_t* wavefronts,
                              const Cigars cigars,
                              int target_k) {
    if (threadIdx.x == 0) {
        edit_cigar_t* const curr_cigar = cigars.get_cigar(blockIdx.x);
        int edit_cigar_idx = 0;
        int k = target_k;
        int d = wavefronts->d;
        const ewf_offset_t* curr_offsets_column = OFFSETS_PTR(wavefronts->offsets_base, d);
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
    ewf_offset_t * const offsets = OFFSETS_PTR(wavefronts->offsets_base, wavefronts->d);

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

        // Accumulate to register, is this really necessary or the compiler
        // would do it for us?
        int acc = 0;
        while ((v + 4) < plen && (h + 4) < tlen) {
            int packed_pattern = 0;
            int packed_text = 0;

            uint32_t p1 = pattern[v] << 24;
            uint32_t p2 = pattern[v + 1] << 16;
            uint32_t p3 = pattern[v + 2] << 8;
            uint32_t p4 = pattern[v + 3];

            packed_pattern = p1 | p2 | p3 | p4;

            uint32_t t1 = text[h] << 24;
            uint32_t t2 = text[h + 1] << 16;
            uint32_t t3 = text[h + 2] << 8;
            uint32_t t4 = text[h + 3];

            packed_text = t1 | t2 | t3 | t4;

            int lz = __clz(packed_pattern ^ packed_text);
            acc += lz / 8;
            if (lz < 32) {
                goto end_extend;
            }

            v += 4;
            h += 4;
        }

        while (v < plen && h < tlen && pattern[v++] == text[h++])
            acc++;

end_extend:
        offsets[k] += acc;
    }
    __syncthreads();
}

__device__ void WF_compute_kernel (edit_wavefronts_t* const wavefronts) {
    ewf_offset_t * const offsets = OFFSETS_PTR(wavefronts->offsets_base, wavefronts->d);
    ewf_offset_t * const next_offsets = OFFSETS_PTR(wavefronts->offsets_base, (wavefronts->d + 1));

    int tid = threadIdx.x;
    int tnum = blockDim.x;

    const int hi = wavefronts->d;
    const int lo = -wavefronts->d;

    if (tid == 0) {
        // Loop peeling (k=lo-1)
        next_offsets[lo - 1] = offsets[lo];
        // Loop peeling (k=lo)
        const ewf_offset_t bottom_upper_del = ((lo+1) <= hi) ? offsets[lo+1] : -1;
        next_offsets[lo] = max(offsets[lo ]+ 1, bottom_upper_del);
    }

    for(int k = lo + tid + 1; k <= hi - 1; k += tnum) {
        const ewf_offset_t max_ins_sub = max(offsets[k], offsets[k - 1]) + 1;
        next_offsets[k] = max(max_ins_sub, offsets[k + 1]);
    }

    if (tid == 0) {
        // Loop peeling (k=hi)
        const ewf_offset_t top_lower_ins = (lo <= (hi-1)) ? offsets[hi-1] : -1;
        next_offsets[hi] = max(offsets[hi],top_lower_ins) + 1;
        // Loop peeling (k=hi+1)
        next_offsets[hi+1] = offsets[hi] + 1;
    }
    __syncthreads();
}

__global__ void WF_edit_distance (const WF_element* elements,
                                  edit_wavefronts_t* const alignments,
                                  const size_t max_distance,
                                  const size_t max_seq_len,
                                  const Cigars cigars) {
    extern __shared__ uint8_t shared_mem_chunk[];
    const int tid = threadIdx.x;
    const WF_element element = elements[blockIdx.x];
    edit_wavefronts_t* wavefronts = &(alignments[blockIdx.x]);
    SEQ_TYPE* shared_text = (SEQ_TYPE*)&shared_mem_chunk[0];
    SEQ_TYPE* shared_pattern = (SEQ_TYPE*)&shared_mem_chunk[max_seq_len];
    int text_len = element.tlen;
    int pattern_len = element.plen;

    // Put text and pattern to shared memory
    // TODO: Separated loops use better the cache?
    for (int i=tid; i<max_seq_len; i += blockDim.x) {
        shared_text[i] = element.text[i];
    }

    for (int i=tid; i<max_seq_len; i += blockDim.x) {
        shared_pattern[i] = element.pattern[i];
    }

    const int target_k = EWAVEFRONT_DIAGONAL(text_len, pattern_len);
    const int target_k_abs = (target_k >= 0) ? target_k : -target_k;
    const ewf_offset_t target_offset = EWAVEFRONT_OFFSET(text_len, pattern_len);
    int distance;

    for (distance = 0; distance < max_distance; distance++) {
        wavefronts->d = distance;
        WF_extend_kernel(element, wavefronts, shared_text, shared_pattern,
                         text_len, pattern_len);
        ewf_offset_t* curr_offsets = OFFSETS_PTR(wavefronts->offsets_base, wavefronts->d);
        if (target_k_abs <= distance && curr_offsets[target_k] == target_offset)
            break;

        WF_compute_kernel(wavefronts);
    }
    WF_backtrace(wavefronts, cigars, target_k);

#ifdef DEBUG_MODE
    if (blockIdx.x == 0 && tid == 0) {
        if (distance == (max_distance - 1)) {
            // TODO
            printf("Max distance reached!!!\n");
        }
        char tmp = element.text[text_len];
        element.text[text_len] = '\0';
        printf("TEXT: %s\n", element.text);
        element.text[text_len] = tmp;
        tmp = element.pattern[pattern_len];
        element.pattern[pattern_len] = '\0';
        printf("PATTERN: %s\n", element.pattern);
        element.pattern[pattern_len] = tmp;
        printf("Distance: %d\n", distance);
    }
#endif
}
