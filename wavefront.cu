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

    // Start the clock for benchmark in DEBUG_MODE
    CLOCK_START()

    // Create offsets into the GPU
    req_memory = this->batch_size * sizeof(edit_wavefronts_t);
    if (req_memory > MAX_GPU_SIZE) {
        WF_ERROR("Required memory is bigger than available memory in GPU");
        return false;
    }
    // Allocate memory for the edit_wavefronts_t structure in GPU
    cudaMalloc((void **) &(this->d_wavefronts), req_memory);
    CUDA_CHECK_ERR;

    // Allocate one big memory chunk in the device for all the offsets
    size_t offsets_size_bytes = OFFSETS_TOTAL_ELEMENTS(this->max_distance) * sizeof(ewf_offset_t);
    size_t total_offsets_size = offsets_size_bytes * this->batch_size;
#ifdef DEBUG_MODE
    if (total_offsets_size >= (1 << 30)) {
        DEBUG("Trying to initialize %.2f MiB per offset (total %.2f GiB).",
              (double)offsets_size_bytes / (1 << 20), (double)total_offsets_size / (1 << 30));
    }
    else if (offsets_size_bytes >= (1 << 20)) {
        DEBUG("Trying to initialize %.2f MiB per offset (total %.2f MiB).",
              (double)offsets_size_bytes / (1 << 20), (double)total_offsets_size / (1 << 20));
    }
    else {
        if (total_offsets_size >= (1 << 20)) {
            DEBUG("Trying to initialize %.2f KiB per offset (total %.2f MiB).",
                  (double)offsets_size_bytes / (1 << 10), (double)total_offsets_size / (1 << 20));
        }
        else {
            DEBUG("Trying to initialize %.2f KiB per offset (total %.2f KiB).",
                  (double)offsets_size_bytes / (1 << 10), (double)total_offsets_size / (1 << 10));
        }
    }
#endif
    // 2 offsets per element (wavefront and next_wavefront)
    cudaMalloc((void **) &base_ptr, total_offsets_size);
    CUDA_CHECK_ERR;
    this->offsets_device_ptr = (ewf_offset_t*)base_ptr;
    cudaMemset((void *) base_ptr, 0, total_offsets_size);
    CUDA_CHECK_ERR;


    // A temporary CPU edit_wavefronts_t array is needed. As we can not access
    // pointers inside GPU, the device pointers results are saved on host RAM,
    // and then send the data to device just once.
    edit_wavefronts_t *tmp_host_wavefronts;
    tmp_host_wavefronts = (edit_wavefronts_t*)calloc(this->batch_size,
                                                     sizeof(edit_wavefronts_t));

    for (int i=0; i<this->batch_size; i++) {
        ewf_offset_t *curr_offset = (ewf_offset_t*)(base_ptr + i * offsets_size_bytes);
        edit_wavefronts_t *curr_host_wf = &tmp_host_wavefronts[i];
        curr_host_wf->offsets_base = curr_offset;
        curr_host_wf->d = 0;
    }

    cudaMemcpy(this->d_wavefronts, tmp_host_wavefronts,
               req_memory, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERR;

    free(tmp_host_wavefronts);

#ifdef DEBUG_MODE
    total_memory = req_memory + total_offsets_size;
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
    cudaFree(this->sequences_device_ptr);
    CUDA_CHECK_ERR;
    cudaFree(this->d_elements);
    CUDA_CHECK_ERR;

    CLOCK_STOP("Text/patterns GPU memory freed.")

    // Free all the offsets
    CLOCK_START()
    cudaFree(this->offsets_device_ptr);
    CUDA_CHECK_ERR;
    cudaFree(this->d_wavefronts);
    CUDA_CHECK_ERR;

    CLOCK_STOP("Offsets GPU memory freed.")

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
    DEBUG("Rearranging memory for batch iteration %d (position %d)", this->batch_idx, curr_position);
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
               this->batch_size * sizeof(WF_element), cudaMemcpyHostToDevice,
               this->HtD_stream);
    CUDA_CHECK_ERR;

    size_t offsets_size_bytes = OFFSETS_TOTAL_ELEMENTS(this->max_distance) * sizeof(ewf_offset_t);
    size_t total_offsets_size = offsets_size_bytes * this->batch_size;
    cudaMemsetAsync(this->offsets_device_ptr, 0, total_offsets_size, this->HtD_stream);
    CUDA_CHECK_ERR

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
    int shared_mem = this->sequences_reader.max_seq_len * 2;

    // Wait until the sequences are copied to the device
    cudaStreamSynchronize(this->HtD_stream);
    CUDA_CHECK_ERR

    DEBUG("Launching wavefront alignment on GPU. %d elements with %d blocks "
          "of %d threads, and %d KiB of shared memory", blocks_x, blocks_x,
          threads_x, shared_mem / (1 << 10));

    WF_edit_distance<<<numBlocks, blockDim, shared_mem>>>(this->d_elements,
                                              this->d_wavefronts,
                                              this->sequences_device_ptr,
                                              this->max_distance,
                                              this->sequences_reader.max_seq_len,
                                              this->d_cigars);
#ifdef DEBUG_MODE
    this->h_cigars.copyIn(this->d_cigars);
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
