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

#include "wavefront.h"
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

// Assumes 1 block per element
// TODO: Offsets are stored in an "array of structs" pattern, it'd be more
// efficient for memory acceses to store them as a "struct of arrays" (for loop)
// ^ is this really true ????
__device__ void WF_extend_kernel (const WF_element element,
                                  edit_wavefronts_t* const wavefronts) {
    ewf_offset_t * const offsets = OFFSETS_PTR(wavefronts->offsets_base, wavefronts->d);

    int tid = threadIdx.x;
    int tnum = blockDim.x;

    const SEQ_TYPE* text = element.text;
    const SEQ_TYPE* pattern = element.pattern;

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
        // This will probably create a lot of divergence ???
        // TODO: Also do this in parallel
        while (v < element.len && h < element.len && pattern[v++] == text[h++])
            acc++;
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
                                  const size_t max_distance) {
    const WF_element element = elements[blockIdx.x];
    edit_wavefronts_t* wavefronts = &(alignments[blockIdx.x]);

    const int target_k = EWAVEFRONT_DIAGONAL(element.len, element.len);
    const int target_k_abs = (target_k >= 0) ? target_k : -target_k;
    const ewf_offset_t target_offset = EWAVEFRONT_OFFSET(element.len, element.len);
    int distance;

    for (distance = 0; distance < max_distance; distance++) {
        wavefronts->d = distance;
        WF_extend_kernel(element, wavefronts);
        ewf_offset_t* curr_offsets = OFFSETS_PTR(wavefronts->offsets_base, wavefronts->d);
        if (target_k_abs <= distance && curr_offsets[target_k] == target_offset)
            break;

        WF_compute_kernel(wavefronts);
    }

#ifdef DEBUG_MODE
    const int tid = threadIdx.x;
    if (blockIdx.x == 0 && tid == 0) {
        if (distance == (max_distance - 1)) {
            // TODO
            printf("Max distance reached!!!\n");
        }
        char tmp = element.text[element.len];
        element.text[element.len] = '\0';
        printf("TEXT: %s\n", element.text);
        element.text[element.len] = tmp;
        tmp = element.pattern[element.len];
        element.pattern[element.len] = '\0';
        printf("PATTERN: %s\n", element.pattern);
        element.pattern[element.len] = tmp;
        printf("Distance: %d\n", distance);
    }
#endif
}
