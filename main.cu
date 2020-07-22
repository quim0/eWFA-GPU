/*;
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

#include <pthread.h>
#include "utils.h"
#include "wavefront.cuh"

const char *USAGE_STR = "Usage:\n"
                        "WFA_edit_gpu <file> <sequence_length> <num_of_sequences> "
                        "<batch_size=num_of_sequences>\n";

struct Parameters {
    char* seq_file;
    size_t seq_len;
    size_t num_sequences;
    size_t batch_size;
    unsigned int tid;
    unsigned int total_threads;
    uint8_t* pinned_mem;
};

void * worker(void * tdata) {
    Parameters* params = (Parameters*)tdata;

    DEBUG("Starting thread %d.", params->tid);

    unsigned int seqs_to_process = params->num_sequences / params->total_threads;
    if (params->tid == (params->total_threads - 1))
        seqs_to_process += params->num_sequences % params->total_threads;

    if (params->batch_size > seqs_to_process) {
        params->batch_size = seqs_to_process;
        // TODO: Create WF warning
        WF_ERROR("[Thread %d] Changing batch size to %d", params->tid, seqs_to_process);
    }

    SequenceReader reader = SequenceReader(params->seq_file, params->seq_len,
                                           seqs_to_process, params->batch_size,
                                           (SEQ_TYPE*)params->pinned_mem);

    reader.skip_n_alignments(seqs_to_process * params->tid);

    if (!reader.read_batch_sequences()) {
        WF_FATAL("[Thread %d] Could not read the sequences from file %s\n", params->tid, params->seq_file);
    }

    Sequences seqs = Sequences(reader.sequences, seqs_to_process,
                               params->seq_len,  params->batch_size, reader);
    seqs.GPU_memory_init();
    do {
        seqs.GPU_launch_wavefront_distance();
    } while (seqs.prepare_next_batch());
    seqs.GPU_memory_free();

    DEBUG("Finished thread %d.", params->tid);

    return NULL;
}

int main (int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        WF_FATAL(USAGE_STR);
    }

    char* seq_file = argv[1];
    size_t seq_len = atoi(argv[2]);
    size_t num_sequences = atoi(argv[3]);
    size_t batch_size = (argc >= 5) ? atoi(argv[4]) : num_sequences;
    unsigned int num_threads = (argc == 6) ? atoi(argv[5]) : 1;

    if (num_threads < 1)
        WF_FATAL("Minimum needed is 1 thread");

    if (batch_size > num_sequences)
        WF_FATAL("batch_size must be >= than the number of alignments.");

    pthread_t *threads = (pthread_t*)calloc(num_threads, sizeof(pthread_t));
    if (threads == NULL)
        WF_FATAL(NOMEM_ERR_STR);

    // Allocate a single chunk of pinned memory
    // *2 --> 2 sequences per alignment
    // *2 --> make sure the sequence fit in memory in the worst case (100% error)
    size_t pinned_mem_per_thread = batch_size * seq_len * 2 * 2 * sizeof(SEQ_TYPE); // Sequences buffer
    size_t total_pinned_mem = pinned_mem_per_thread * num_threads;
    uint8_t* pinned_mem;
    cudaMallocHost(&pinned_mem, total_pinned_mem);
    CUDA_CHECK_ERR
    if (!pinned_mem)
        WF_FATAL(NOMEM_ERR_STR)
    
    Parameters *params_array = (Parameters*)calloc(num_threads, sizeof(Parameters));
    
    for (unsigned int i=1; i<num_threads; i++) {
        Parameters *params = &params_array[i];
        params->seq_file = seq_file;
        params->seq_len = seq_len;
        params->num_sequences = num_sequences;
        params->batch_size = batch_size;
        params->tid = i;
        params->total_threads = num_threads;
        params->pinned_mem = pinned_mem + (pinned_mem_per_thread * i);
        DEBUG("Launching thread %d", params->tid);
        if (pthread_create(&threads[i], NULL, worker, params))
            WF_FATAL("Can not create threads.");
    }

    Parameters *params = &params_array[0];
    params->seq_file = seq_file;
    params->seq_len = seq_len;
    params->num_sequences = num_sequences;
    params->batch_size = batch_size;
    params->tid = 0;
    params->total_threads = num_threads;
    params->pinned_mem = pinned_mem;
    worker(params);

    for (int i=1; i<num_threads; i++) {
        // tid is i+1 because thread 0 is the thread that launches the others
        if (pthread_join(threads[i], NULL))
            WF_FATAL("Can not join thread.");
    }

    free(params_array);
    free(threads);

    return 0;
}
