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

#define EWAVEFRONT_V(k,offset) ((offset)-(k))
#define EWAVEFRONT_H(k,offset) (offset)

#define WRAP_SIZE 32

// Assumes 1 block per element
__global__ void WF_extend_kernel (const WF_element* elements,
                                  edit_wavefronts_t* const wavefronts) {
    const WF_element element = elements[blockIdx.x];
    edit_wavefront_t * const wavefront = &(wavefronts[blockIdx.x].wavefront);
    ewf_offset_t * const offsets = wavefront->offsets;

    int tid = threadIdx.x;
    int tnum = blockDim.x;

    const int k_min = wavefront->lo;
    const int k_max = wavefront->hi;
    for (int k = k_min; k<=k_max; k++) {
        const int v_base  = EWAVEFRONT_V(k, offsets[k]);
        const int h_base  = EWAVEFRONT_H(k, offsets[k]);
        int v = v_base + tid;
        int h = h_base + tid;
        __shared__ unsigned int to_sum;
        if (tid == 0)
            to_sum = 0xffffffff;
        __syncthreads();
        // TODO: Don't iterate over all the text/pattern, find a safe way to
        // stop other threads when one finds a missmatch
        // TODO: Do the minimum at wrap level and make only one atomic operation
        while (v < element.len && h < element.len) {
            bool equal = element.pattern[v] == element.text[h];

            if (!equal) {
                int curr_position = v - v_base;
                atomicMin(&to_sum, curr_position);
                break;
            }

            v += tnum;
            h += tnum;
        }
        __syncthreads();

        offsets[k] += to_sum;
    }
}

__global__ void WF_compute (WF_element* element) {
    
}
