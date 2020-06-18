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
#include "wavefront.h"

// TODO: Check max_gpu_size to take in accont the multiple arrays stored on GPU
bool Sequences::GPU_memory_init () {
    CLOCK_INIT()
    // Send patterns to GPU
    size_t req_memory = this->num_elements * sizeof(WF_element);
    if (req_memory > MAX_GPU_SIZE) {
        WF_ERROR("Required memory is bigger than available memory in GPU");
        return false;
    }

    // Start the clock for benchmanrk purposes if DEBUG_MODE is enabled
    CLOCK_START()

    cudaMalloc((void **) &(this->d_elements), req_memory);
    CUDA_CHECK_ERR;

    // Allocate a big chunk of memory only once for the different patterns and
    // texts
    uint8_t* base_ptr;
    size_t seq_size_bytes = this->sequence_len * sizeof(SEQ_TYPE);
    // *2 sequences per element (pattern and text)
    cudaMalloc((void **) &(base_ptr),
               (seq_size_bytes * 2) * this->num_elements);
    CUDA_CHECK_ERR;
    DEBUG("Allocating memory to store text/patterns on device (%zu bytes)",
          seq_size_bytes * 2 * this->num_elements);

    // Copy all the secuences to the fresh allocated memory on GPU
    cudaMemcpy(base_ptr, this->elements[0].text,
               (seq_size_bytes * 2) * this->num_elements,
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERR;

    // Create a temporary array o WF_elements in host memory, store all the
    // device pointers here and then do a single memcpy. This is done to avoid
    // doing a cudaMemcpy on each iteration of the loop, which dramatically
    // drops the performance.
    WF_element *tmp_wf_elements_host = (WF_element*)
                                calloc(this->num_elements, sizeof(WF_element));

    // += 2 because every element have two sequences (pattern and text)
    for (int i=0; i<(this->num_elements * 2); i += 2) {
        WF_element *tmp_host_elem = &tmp_wf_elements_host[i / 2];
        SEQ_TYPE* seq1 = (SEQ_TYPE*)(base_ptr + i * seq_size_bytes);
        SEQ_TYPE* seq2 = (SEQ_TYPE*)(base_ptr + (i + 1) * seq_size_bytes);
        tmp_host_elem->text = seq1;
        tmp_host_elem->pattern = seq2;
        tmp_host_elem->len = this->sequence_len;
    }
    cudaMemcpy(this->d_elements, tmp_wf_elements_host,
               this->num_elements * sizeof(WF_element), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERR;
    free(tmp_wf_elements_host);

#ifdef DEBUG_MODE
    size_t total_memory  = req_memory + seq_size_bytes * 2 * this->num_elements;
    DEBUG("GPU pattern/text memory initialized, %zu MiB used.",
          total_memory / (1 << 20));
    CLOCK_STOP("GPU pattern/text memory initializaion.")
#endif

    // Start the clock for benchmark in DEBUG_MODE
    CLOCK_START()

    // Create offsets into the GPU
    req_memory = this->num_elements * sizeof(edit_wavefronts_t);
    if (req_memory > MAX_GPU_SIZE) {
        WF_ERROR("Required memory is bigger than available memory in GPU");
        return false;
    }
    // Allocate memory for the edit_wavefronts_t structure in GPU
    cudaMalloc((void **) &(this->d_wavefronts), req_memory);
    CUDA_CHECK_ERR;
    cudaMemset((void *) this->d_wavefronts, 0, req_memory);
    CUDA_CHECK_ERR;

    // Allocate one big memory chunk in the device for all the offsets
    size_t offset_size_bytes = 2 * this->max_distance * sizeof(ewf_offset_t);
    // 2 offsets per element (wavefront and next_wavefront)
    cudaMalloc((void **) &base_ptr, offset_size_bytes * this->num_elements * 2);
    CUDA_CHECK_ERR;
    cudaMemset((void *) base_ptr, 0, offset_size_bytes * this->num_elements * 2);
    CUDA_CHECK_ERR;

    // A temporary CPU edit_wavefronts_t array is needed. As we can not access
    // pointers inside GPU, the device pointers results are saved on host RAM,
    // and then send the data to device just once.
    edit_wavefronts_t *tmp_host_wavefronts = (edit_wavefronts_t*)calloc(
                                this->num_elements, sizeof(edit_wavefronts_t));

    // += 2 because every element have two offsets (wavefront and next_wavefront)
    for (int i=0; i<(this->num_elements * 2); i += 2) {
        ewf_offset_t *offset1 = (ewf_offset_t*)(base_ptr + i * offset_size_bytes + this->max_distance);
        ewf_offset_t *offset2 = (ewf_offset_t*)(base_ptr + (i + 1) * offset_size_bytes + this->max_distance);
        edit_wavefronts_t *curr_host_wf = &(tmp_host_wavefronts[i/2]);
        curr_host_wf->wavefront.offsets = offset1;
        curr_host_wf->next_wavefront.offsets = offset2;
    }

    cudaMemcpy(this->d_wavefronts, tmp_host_wavefronts,
               req_memory, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERR;

    free(tmp_host_wavefronts);

#ifdef DEBUG_MODE
    total_memory = req_memory + offset_size_bytes * this->num_elements * 2;
    DEBUG("GPU offsets memory initialized, %zu MiB used.", total_memory / (1 << 20));
    CLOCK_STOP("GPU offsets memory initialization.")
#endif

    return true;
}

bool Sequences::GPU_memory_free () {
    DEBUG("Freeing GPU memory.");
    CLOCK_INIT()
    CLOCK_START()
    // Free all the texts/patterns
    WF_element tmp_host_elem = {0};
    cudaMemcpy(&tmp_host_elem, this->d_elements,
               sizeof(WF_element), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERR;
    cudaFree(tmp_host_elem.text);
    CUDA_CHECK_ERR;
    cudaFree(this->d_elements);
    CUDA_CHECK_ERR;

    CLOCK_STOP("Text/patterns GPU memory freed.")

    // Free all the offsets
    CLOCK_START()
    edit_wavefront_t tmp_host_wf = {0};
    cudaMemcpy(&tmp_host_wf, &this->d_wavefronts[0].wavefront,
               sizeof(edit_wavefront_t), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERR;
    cudaFree(tmp_host_wf.offsets - this->max_distance);
    CUDA_CHECK_ERR;
    cudaFree(d_wavefronts);
    CUDA_CHECK_ERR;

    CLOCK_STOP("Offsets GPU memory freed.")

    DEBUG("GPU memory freed.")
    return true;
}
