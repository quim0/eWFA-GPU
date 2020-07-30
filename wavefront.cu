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
#include "kernels.cuh"

#define EWAVEFRONT_DIAGONAL(h,v) ((h)-(v))
#define EWAVEFRONT_OFFSET(h,v)   (h)

// TODO: Check max_gpu_size to take in accont the multiple arrays stored on GPU
bool Sequences::GPU_memory_init () {
    CLOCK_INIT()
    // Send patterns to GPU
    size_t req_memory = this->batch_size * sizeof(WF_element);
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
    SEQ_TYPE* base_ptr;
    size_t seq_size_bytes = this->sequences_reader.max_seq_len * sizeof(SEQ_TYPE);
    // * 2 sequences per element (pattern and text)
    cudaMalloc((void **) &(base_ptr),
               (seq_size_bytes * 2) * this->batch_size);
    CUDA_CHECK_ERR;
    this->sequences_device_ptr = base_ptr;
    DEBUG("Allocating memory to store text/patterns on device (%zu MiB)",
          (seq_size_bytes * 2 * this->batch_size) / (1 << 20));

    // Copy all the secuences for this batch to the fresh allocated memory on
    // GPU
    SEQ_TYPE* initial_seq_ptr = this->sequences_reader.get_sequences_buffer() +
                                (this->sequences_reader.max_seq_len * 2 *
                                this->initial_alignment);
    cudaMemcpy(base_ptr, initial_seq_ptr,
               (seq_size_bytes * 2) * this->batch_size,
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERR;

    cudaMemcpy(this->d_elements, &this->elements[this->initial_alignment],
               this->batch_size * sizeof(WF_element), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERR;

#ifdef DEBUG_MODE
    size_t total_memory  = req_memory + seq_size_bytes * 2 * this->batch_size;
    DEBUG("GPU pattern/text memory initialized, %zu MiB used.",
          total_memory / (1 << 20));
    CLOCK_STOP("GPU pattern/text memory initializaion.")
#endif

    return true;
}

bool Sequences::GPU_memory_free () {
    DEBUG("Freeing GPU memory.");
    CLOCK_INIT()
    CLOCK_START()
    // Free all the texts/patterns
    cudaFree(this->sequences_device_ptr);
    CUDA_CHECK_ERR;
    cudaFree(this->d_elements);
    CUDA_CHECK_ERR;

    CLOCK_STOP("Text/patterns GPU memory freed.")

    DEBUG("GPU memory freed.")
    return true;
}

#if 0
bool Sequences::CPU_read_next_sequences () {
    DEBUG("Reading next batch elements from the sequences file.");
    return this->sequences_reader.read_batch_sequences();
}
#endif

bool Sequences::GPU_prepare_memory_next_batch () {
    // first "alginment" (sequence pair) of the current batch
    int curr_position = (++this->batch_idx * this->batch_size);
    if (curr_position >= this->num_elements) {
        DEBUG("All batches have already been processed.");
        return false;
    }
    DEBUG("Rearranging memory for batch iteration %d (position %d)",
           this->batch_idx, initial_alignment + curr_position);
    // The last "batch" may be sorter than a complete batch, e.g 10 elements,
    // batch size of 3
    int curr_batch_size = ((this->num_elements - curr_position) < this->batch_size) ?
                            (this->num_elements - curr_position) : this->batch_size;

    // Send the new text/pattern sequences to device
    size_t seq_size_bytes = this->sequences_reader.max_seq_len * sizeof(SEQ_TYPE);
    SEQ_TYPE* first_pos_ptr = TEXT_PTR(this->elements[curr_position + this->initial_alignment].alignment_idx,
                                   this->sequences_reader.get_sequences_buffer(),
                                   this->sequences_reader.max_seq_len);
    cudaMemcpyAsync(this->sequences_device_ptr,
               first_pos_ptr,
               (seq_size_bytes * 2) * curr_batch_size,
               cudaMemcpyHostToDevice,
               this->HtD_stream);
    CUDA_CHECK_ERR

    // Send the new text_len and pattern_len to device
    cudaMemcpyAsync(this->d_elements, &this->elements[curr_position + initial_alignment],
               curr_batch_size * sizeof(WF_element), cudaMemcpyHostToDevice,
               this->HtD_stream);
    CUDA_CHECK_ERR;

    return true;
}

void Sequences::GPU_launch_wavefront_distance () {
    // TODO: Determine better the number of threads
    int threads_x = 32;

    int blocks_x;
    // Check if the current batch is smaller than "batch_size"
    size_t sequences_remaining = this->num_elements - this->batch_size *
                                 this->batch_idx;
    if (sequences_remaining < this->batch_size)
        blocks_x = sequences_remaining;
    else
        blocks_x = this->batch_size;
    blocks_x = (blocks_x > MAX_BLOCKS) ?  MAX_BLOCKS : blocks_x;

    dim3 numBlocks(blocks_x, 1);
    dim3 blockDim(threads_x, 1);

    // text + pattern with allowance of 100% error
    int shared_mem = this->sequences_reader.max_seq_len * 2
                     // 2 complete wavefronts, add 2 to the number of elements
                     // in a wavefront to avoid loop peeling
                     + 2 * (WF_ELEMENTS(this->max_distance) + 2) * sizeof(ewf_offset_t);

    // Wait until the sequences are copied to the device
    cudaStreamSynchronize(this->HtD_stream);
    CUDA_CHECK_ERR

    DEBUG("Launching wavefront alignment on GPU. %d elements with %d blocks "
          "of %d threads, and %d KiB of shared memory", blocks_x, blocks_x,
          threads_x, shared_mem / (1 << 10));

    WF_edit_distance<<<numBlocks, blockDim, shared_mem>>>(this->d_elements,
                                              this->sequences_device_ptr,
                                              this->max_distance,
                                              this->sequences_reader.max_seq_len,
                                              this->d_cigars);
#ifdef DEBUG_MODE
    // CopyIn copies the packed backtraces in
    this->h_cigars.copyIn(this->d_cigars);
    //size_t bt_offset_results = this->batch_size * WF_ELEMENTS(max_distance) * sizeof(WF_backtrace_t);
    //cudaMemcpy((void*)this->backtraces_host_ptr, this->result_backtraces_device_ptr,
    //           this->batch_size * sizeof(WF_backtrace_t), cudaMemcpyDeviceToHost);
    //CUDA_CHECK_ERR;
    size_t curr_position = (this->batch_idx * this->batch_size) +
                        this->initial_alignment;
    SEQ_TYPE* seq_base_ptr = this->sequences_reader.get_sequences_buffer();
    size_t max_seq_len = this->sequences_reader.max_seq_len;
    int total_corrects = 0;
    for (int i=0; i<blocks_x; i++)
        if (this->h_cigars.check_cigar(i, this->elements[curr_position + i], seq_base_ptr, max_seq_len))
            total_corrects++;

    if (total_corrects == blocks_x)
        DEBUG_GREEN("Correct alignments: %d/%d", total_corrects, blocks_x)
    else
        DEBUG_RED("Correct alignments: %d/%d", total_corrects, blocks_x)
#endif
}

// Returns false when everything is comple
bool Sequences::prepare_next_batch () {
    // Wait for the kernel to finish
    cudaStreamSynchronize(0);
    CUDA_CHECK_ERR;
    bool ret;


    // This is async
    ret = this->GPU_prepare_memory_next_batch();

    // This is sync
    this->h_cigars.copyIn(this->d_cigars);

    // Put all the device cigars at 0 again
    this->d_cigars.device_reset();

    return ret;
}
