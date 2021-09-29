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
#include "kernels.cuh"

#define EWAVEFRONT_DIAGONAL(h,v) ((h)-(v))
#define EWAVEFRONT_OFFSET(h,v)   (h)

// TODO: Check max_gpu_size to take in accont the multiple arrays stored on GPU
bool Sequences::GPU_memory_init () {
    CLOCK_INIT()

    // Start the clock for benchmanrk purposes if DEBUG_MODE is enabled
    CLOCK_START()

    size_t req_memory = this->batch_size * sizeof(WF_element);
    cudaMalloc((void **) &(this->d_elements), req_memory);
    CUDA_CHECK_ERR;

    // Allocate a big chunk of memory only once for the different packed
    // patterns and texts
    SEQ_TYPE* base_ptr;
    size_t seq_size_bytes = this->sequences_reader.max_seq_len * sizeof(SEQ_TYPE);
    // * 2 sequences per element (pattern and text)
    cudaMalloc((void **) &(base_ptr),
               (seq_size_bytes * 2) * this->batch_size);
    CUDA_CHECK_ERR;
    cudaMemset(base_ptr, 0, (seq_size_bytes * 2) * this->batch_size);
    CUDA_CHECK_ERR;
    this->sequences_device_ptr = base_ptr;
    DEBUG("Allocating memory to store packed text/patterns on device (%zu MiB)",
          (seq_size_bytes * 2 * this->batch_size) / (1 << 20));


    // Allocate a big chunk of memory only once for the different patterns and
    // texts
    SEQ_TYPE* base_ptr_unpacked;
    seq_size_bytes = this->sequences_reader.max_seq_len_unpacked * sizeof(SEQ_TYPE);
    // * 2 sequences per element (pattern and text)
    cudaMalloc((void **) &(base_ptr_unpacked),
               (seq_size_bytes * 2) * this->batch_size);
    CUDA_CHECK_ERR;
    this->sequences_device_ptr_unpacked = base_ptr_unpacked;
    DEBUG("Allocating memory to store unpacked text/patterns on device (%zu MiB)",
          (seq_size_bytes * 2 * this->batch_size) / (1 << 20));

    // Copy all the unpacked secuences for this batch to the fresh allocated
    // memory on GPU
    SEQ_TYPE* initial_seq_ptr =
            this->sequences_reader.get_sequences_buffer_unpacked() +
            (this->sequences_reader.max_seq_len_unpacked * 2 * this->initial_alignment);
    cudaMemcpy(base_ptr_unpacked, initial_seq_ptr,
               (seq_size_bytes * 2) * this->batch_size,
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERR;

    dim3 gridSize(this->batch_size);
    // Max block size is 1024 threads, so, even if the sequence length is more
    // than 512, we are limited to use 512 threads for pattern, and another 512
    // for the text.
    dim3 blockSize(min(this->sequences_reader.seq_len/4, 256L), 2);
    DEBUG("Lauching sequences packing kernel on GPU with grid(%d) and block(%d, %d).",
          gridSize.x, blockSize.x, blockSize.y);
    // Unpacked pattern+text in shared memory
    int shmem_size = this->sequences_reader.max_seq_len_unpacked * 2;
    compact_sequences<<<gridSize, blockSize, shmem_size>>>(
                                        this->sequences_device_ptr_unpacked,
                                        this->sequences_device_ptr,
                                        this->sequences_reader.max_seq_len_unpacked,
                                        this->sequences_reader.max_seq_len);

    CUDA_CHECK_ERR;

    cudaMemcpyAsync(this->d_elements, &this->elements[this->initial_alignment],
               this->batch_size * sizeof(WF_element), cudaMemcpyHostToDevice, 0);
    CUDA_CHECK_ERR;

    cudaStreamSynchronize(0);
    DEBUG("Sequences packed.");

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
    cudaFree(this->sequences_device_ptr_unpacked);
    CUDA_CHECK_ERR;
    cudaFree(this->d_elements);
    CUDA_CHECK_ERR;

    CLOCK_STOP("Text/patterns GPU memory freed.")

    DEBUG("GPU memory freed.")
    return true;
}

bool Sequences::GPU_prepare_memory_next_batch () {
    // first "alginment" (sequence pair) of the current batch
    int curr_position = ((this->batch_idx + 1) * this->batch_size);
    // This is redundant but just in case
    if (curr_position >= this->num_elements) {
        DEBUG("All batches have already been processed.");
        return false;
    }
    DEBUG("Rearranging memory for batch iteration %d (position %d)",
           this->batch_idx+1, initial_alignment + curr_position);
    // The last "batch" may be sorter than a complete batch, e.g 10 elements,
    // batch size of 3
    int curr_batch_size = ((this->num_elements - curr_position) < this->batch_size) ?
                            (this->num_elements - curr_position) : this->batch_size;

    // Zero the packed sequences
    // TODO: Is this really necessary, as we have the distance
    cudaMemsetAsync(this->sequences_device_ptr,
                    0,
                    this->sequences_reader.max_seq_len * 2 * this->batch_size,
                    this->HtD_stream);


    dim3 gridSize(this->batch_size);
    // Max block size is 1024 threads, so, even if the sequence length is more
    // than 512, we are limited to use 512 threads for pattern, and another 512
    // for the text.
    dim3 blockSize(min(this->sequences_reader.seq_len/4, 256L), 2);
    // Pack the sequences on GPU
    int shmem_size = this->sequences_reader.max_seq_len_unpacked * 2;
    compact_sequences<<<gridSize, blockSize, shmem_size, this->HtD_stream>>>(
                                        this->sequences_device_ptr_unpacked,
                                        this->sequences_device_ptr,
                                        this->sequences_reader.max_seq_len_unpacked,
                                        this->sequences_reader.max_seq_len);

    CUDA_CHECK_ERR;
    // Send the new text_len and pattern_len to device, this is launched in
    // stream 0 so it can be overlapped with compact_sequences kernel
    cudaMemcpyAsync(this->d_elements, &this->elements[curr_position + initial_alignment],
               curr_batch_size * sizeof(WF_element), cudaMemcpyHostToDevice, 0);
    CUDA_CHECK_ERR;

    return true;
}

void Sequences::GPU_launch_wavefront_distance () {
    int threads_x = THREADS_PER_BLOCK;

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

    // Wait until the sequences are packed in the device
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
}

// Returns false when everything is comple
bool Sequences::prepare_next_batch (bool print_cigars) {
    // Async, memset(0) the ascii CIGARs buffer on the device
    this->d_cigars.device_reset_ascii();

    int curr_position = (this->batch_idx * this->batch_size);
    int curr_batch_size = \
        ((this->num_elements - curr_position) < this->batch_size) ?
        (this->num_elements - curr_position) : this->batch_size;

    // Use at most 50KiB of shared memory
    int ALIGNMENTS_PER_BLOCK;
    bool use_shared_mem;
    if (this->sequences_reader.max_seq_len_unpacked < 400) {
        ALIGNMENTS_PER_BLOCK = 128;
        use_shared_mem = true;
    } else if (this->sequences_reader.max_seq_len_unpacked < 800) {
        ALIGNMENTS_PER_BLOCK = 64;
        use_shared_mem = true;
    } else if (this->sequences_reader.max_seq_len_unpacked < 1600) {
        ALIGNMENTS_PER_BLOCK = 32;
        use_shared_mem = true;
    } else {
        ALIGNMENTS_PER_BLOCK = 512;
        use_shared_mem = false;
    }

    // Blocks of 512 threads (arbitrary choice)
    int num_blocks = curr_batch_size / ALIGNMENTS_PER_BLOCK;
    num_blocks = (curr_batch_size % ALIGNMENTS_PER_BLOCK) ? num_blocks+1 : num_blocks;
    dim3 gridSize(num_blocks, 1);
    dim3 blockSize(ALIGNMENTS_PER_BLOCK, 1);

    size_t sh_mem_size = ALIGNMENTS_PER_BLOCK * this->h_cigars.cigar_max_len;

    // This is async, but it's on the same stream than the alignment kernel, so
    // it won't execute until the alignment kernel has finished.
    if (use_shared_mem) {
        generate_cigars_sh_mem<<<gridSize, blockSize, sh_mem_size>>>(
                              this->sequences_device_ptr,
                              this->d_elements,
                              this->sequences_reader.max_seq_len,
                              this->d_cigars,
                              curr_batch_size);
    } else {
        generate_cigars<<<gridSize, blockSize>>>(this->sequences_device_ptr,
                              this->d_elements,
                              this->sequences_reader.max_seq_len,
                              this->d_cigars,
                              curr_batch_size);
    }

    // Wait for the kernel to finish
    bool finished = this->is_last_iter();

    // Copy next unpacked sequences while the alignment or cigar recovery is
    // executing.
    if (!finished) {
        int next_position = ((this->batch_idx + 1) * this->batch_size);
        int next_batch_size = \
            ((this->num_elements - next_position) < this->batch_size) ?
            (this->num_elements - next_position) : this->batch_size;
        size_t seq_size_bytes =
            this->sequences_reader.max_seq_len_unpacked * sizeof(SEQ_TYPE);
        SEQ_TYPE* first_pos_ptr = PATTERN_PTR(
            this->elements[next_position + this->initial_alignment].alignment_idx,
            this->sequences_reader.get_sequences_buffer_unpacked(),
            this->sequences_reader.max_seq_len_unpacked);
        cudaMemcpyAsync(this->sequences_device_ptr_unpacked,
                   first_pos_ptr,
                   (seq_size_bytes * 2) * next_batch_size,
                   cudaMemcpyHostToDevice,
                   this->HtD_stream);
        CUDA_CHECK_ERR
    }

    // This is async on stream 0, copy ASCII cigars from device to host
    this->h_cigars.copyInAscii(this->d_cigars);

    // All ASCII cigars are on host, we can start next iteration
    cudaStreamSynchronize(0);
    CUDA_CHECK_ERR;

    if (print_cigars) {
        for (int i=0; i<curr_batch_size; i++) {
            const int curr_position = this->batch_idx * this->batch_size;
            edit_cigar_t* curr_cigar = this->h_cigars.generate_ascii_cigar(i);
            printf("%d: %s\n", this->initial_alignment + curr_position + i, curr_cigar);
            free(curr_cigar);
        }
    }

#ifdef DEBUG_MODE
    size_t curr_alignment = (this->batch_idx * this->batch_size) +
                        this->initial_alignment;
    SEQ_TYPE* seq_base_ptr = this->sequences_reader.get_sequences_buffer_unpacked();
    size_t max_seq_len = this->sequences_reader.max_seq_len_unpacked;
    int total_corrects = 0;
    for (int i=0; i<curr_batch_size; i++) {
        edit_cigar_t* curr_cigar = this->h_cigars.get_cigar_ascii(i);
        if (this->h_cigars.check_cigar(i, this->elements[curr_alignment + i],
            seq_base_ptr, max_seq_len, curr_cigar)) {
            total_corrects++;
        }
    }

    if (total_corrects == curr_batch_size)
        DEBUG_GREEN("Correct alignments: %d/%d", total_corrects, curr_batch_size)
    else
        DEBUG_RED("Correct alignments: %d/%d", total_corrects, curr_batch_size)
#endif

    if (!finished) {
        // This is async
        this->GPU_prepare_memory_next_batch();
    }

    this->batch_idx++;

    return !finished;
}
