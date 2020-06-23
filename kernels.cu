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

#define EWAVEFRONT_V(k,offset) ((offset)-(k))
#define EWAVEFRONT_H(k,offset) (offset)
#define EWAVEFRONT_DIAGONAL(h,v) ((h)-(v))
#define EWAVEFRONT_OFFSET(h,v)   (h)

// TODO: Search instrinsic to use
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

__host__ __device__ void pprint_wavefront(const edit_wavefront_t* const wf) {
    const ewf_offset_t* const offsets = wf->offsets;
    const int hi = wf->hi;
    const int lo = wf->lo;

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
                                  edit_wavefront_t* const wavefront) {
    ewf_offset_t * const offsets = wavefront->offsets;

    int tid = threadIdx.x;
    int tnum = blockDim.x;

    const SEQ_TYPE* text = element.text;
    const SEQ_TYPE* pattern = element.pattern;

    const int k_min = wavefront->lo;
    const int k_max = wavefront->hi;

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

__device__ void WF_compute_kernel (edit_wavefront_t* const wavefront,
                                   edit_wavefront_t* const next_wavefront) {
    ewf_offset_t * const offsets = wavefront->offsets;
    ewf_offset_t * const next_offsets = next_wavefront->offsets;

    int tid = threadIdx.x;
    int tnum = blockDim.x;

    const int lo = wavefront->lo;
    const int hi = wavefront->hi;

    if (tid == 0) {
        next_wavefront->hi = hi + 1;
        next_wavefront->lo = lo - 1;
    }

    __syncthreads();

    // Assume the offsets arrays are memset to -1, so we don't need loop peeling
    for(int k = lo + tid - 1; k <= hi + 1; k += tnum) {
        // TODO: Look at __vmaxu2 cuda intrinsic (works on 16bit packed uints)
        //       there's also a signed version (offsets are signed)

        const ewf_offset_t max_ins_sub = MAX(offsets[k], offsets[k - 1]) + 1;
        next_offsets[k] = MAX(max_ins_sub, offsets[k + 1]);
    }
    __syncthreads();
}

__global__ void WF_edit_distance (const WF_element* elements,
                                  edit_wavefronts_t* const wavefronts) {
    const WF_element element = elements[blockIdx.x];
    edit_wavefront_t* wavefront = &(wavefronts[blockIdx.x].wavefront);
    edit_wavefront_t* next_wavefront = &(wavefronts[blockIdx.x].next_wavefront);

    const int tid = threadIdx.x;

    // Offsets are initialzied to -1, but the initial position must be 0
    if (tid == 0)
        wavefront->offsets[0] = 0;
        wavefront->hi = 0;
        wavefront->lo = 0;
        next_wavefront->hi= 0;
        next_wavefront->lo = 0;

    __syncthreads();

    const int target_k = EWAVEFRONT_DIAGONAL(element.len, element.len);
    const int target_k_abs = (target_k >= 0) ? target_k : -target_k;
    const ewf_offset_t target_offset = EWAVEFRONT_OFFSET(element.len, element.len);
    const int max_distance = element.len * 2;
    int distance;

    for (distance = 0; distance < max_distance; ++distance) {
        WF_extend_kernel(element, wavefront);
        if (target_k_abs <= distance && wavefront->offsets[target_k] == target_offset)
            break;

        WF_compute_kernel(wavefront, next_wavefront);

        edit_wavefront_t* tmp_wavefront = wavefront;
        wavefront = next_wavefront;
        next_wavefront = tmp_wavefront;
    }

#ifdef DEBUG_MODE
    if (blockIdx.x == 0 && tid == 0) {
        char tmp = element.text[element.len];
        element.text[element.len] = '\0';
        printf("TEXT: %s\n", element.text);
        element.pattern[element.len] = '\0';
        element.text[element.len] = tmp;
        printf("PATTERN: %s\n", element.pattern);
        printf("Distance: %d\n", distance);
    }
#endif
}
