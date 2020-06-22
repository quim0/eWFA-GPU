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
// TODO: Offsets are stored in an "array of structs" pattern, it'd be more
// efficient for memory acceses to store them as a "struct of arrays" (for loop)
// ^ is this really true ????
__global__ void WF_extend_kernel (const WF_element* elements,
                                  edit_wavefronts_t* const wavefronts) {
    const WF_element element = elements[blockIdx.x];
    edit_wavefront_t * const wavefront = &(wavefronts[blockIdx.x].wavefront);
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
        __syncthreads();
    }
}

__global__ void WF_compute (WF_element* element) {
    
}
