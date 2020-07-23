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
                        "WFA_edit_gpu <file> <sequence_length> <num_of_aligments> "
                        "<batch_size=num_of_sequences>\n";

struct Parameters {
    char* seq_file;
    size_t seq_len;
    size_t num_alignments;
    size_t batch_size;
    unsigned int tid;
    unsigned int total_threads;
    const SequenceReader *reader;
};

void * worker(void * tdata) {
    Parameters* params = (Parameters*)tdata;

    DEBUG("Starting thread %d.", params->tid);

    unsigned int alignments_to_process = params->num_alignments / params->total_threads;
    unsigned int alignments_to_skip = alignments_to_process * params->tid;
    if (params->tid == (params->total_threads - 1))
        alignments_to_process += params->num_alignments % params->total_threads;

    if (params->batch_size > alignments_to_process) {
        params->batch_size = alignments_to_process;
        // TODO: Create WF warning
        WF_ERROR("[Thread %d] Changing batch size to %d", params->tid, alignments_to_process);
    }

    const SequenceReader *reader = params->reader;


    Sequences seqs = Sequences(reader->sequences, alignments_to_process,
                               params->seq_len,  params->batch_size, *reader,
                               alignments_to_skip);
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
    size_t num_alignments = atoi(argv[3]);
    size_t batch_size = (argc >= 5) ? atoi(argv[4]) : num_alignments;
    unsigned int num_threads = (argc == 6) ? atoi(argv[5]) : 1;

    if (num_threads < 1)
        WF_FATAL("Minimum needed is 1 thread");

    if (batch_size > num_alignments)
        WF_FATAL("batch_size must be >= than the number of alignments.");

    pthread_t *threads = (pthread_t*)calloc(num_threads, sizeof(pthread_t));
    if (threads == NULL)
        WF_FATAL(NOMEM_ERR_STR);

    Parameters *params_array = (Parameters*)calloc(num_threads, sizeof(Parameters));

    SequenceReader reader = SequenceReader(seq_file, seq_len, num_alignments,
                                           batch_size);
    if (!reader.read_file()) {
        WF_FATAL("Could not read the sequences from file %s\n", seq_file);
    }
    
    for (unsigned int i=1; i<num_threads; i++) {
        Parameters *params = &params_array[i];
        params->seq_file = seq_file;
        params->seq_len = seq_len;
        params->num_alignments= num_alignments;
        params->batch_size = batch_size;
        params->tid = i;
        params->total_threads = num_threads;
        params->reader = &reader;
        DEBUG("Launching thread %d", params->tid);
        if (pthread_create(&threads[i], NULL, worker, params))
            WF_FATAL("Can not create threads.");
    }

    Parameters *params = &params_array[0];
    params->seq_file = seq_file;
    params->seq_len = seq_len;
    params->num_alignments = num_alignments;
    params->batch_size = batch_size;
    params->tid = 0;
    params->total_threads = num_threads;
    params->reader = &reader;
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
