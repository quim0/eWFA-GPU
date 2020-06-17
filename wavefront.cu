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

#include "utils.h"
#include "wavefront.cuh"

bool Wavefronts::GPU_memory_init () {
    size_t req_memory = this->num_elements * sizeof(this->WF_element);
    if (req_memory > MAX_GPU_SIZE) {
        std::cout << "Required memory is bigger than available memory in GPU" << std::endl;
        return false;
    }

    cudaMalloc((void **) &(this->d_elements), req_memory);
    CUDA_CHECK_ERR;
    cudaMemcpy(this->d_elements, this->elements, req_memory, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERR;
    DEBUG("GPU memory initialized, %d bytes used.", req_memory)
    return true;
}
