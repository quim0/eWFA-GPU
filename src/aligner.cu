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

#include <pthread.h>
#include "utils/arg_handler.h"
#include "utils/sequence_reader.h"
#include "wavefront.cuh"

#define NUM_ARGUMENTS 6

struct Parameters {
    char* seq_file;
    size_t seq_len;
    size_t num_alignments;
    size_t batch_size;
    bool print_cigars;
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
    } while (seqs.prepare_next_batch(params->print_cigars));
    seqs.GPU_memory_free();

    DEBUG("Finished thread %d.", params->tid);

    return NULL;
}

int main (int argc, char** argv) {

    option_t options_arr[NUM_ARGUMENTS] = {
        // 0
        {.name = "Sequences file",
         .description = "File containing the sequences to align.",
         .short_arg = 'f',
         .long_arg = "file",
         .required = true,
         .type = ARG_STR
         },
        // 1
        {.name = "Number of alignments",
         .description = "Number of alignments to read from the file (default=all"
                        " alignments)",
         .short_arg = 'n',
         .long_arg = "num-alignments",
         .required = true,
         .type = ARG_INT
         },
        // 2
        {.name = "Sequence length",
         .description = "Maximum sequence length.",
         .short_arg = 'l',
         .long_arg = "seq-len",
         .required = true,
         .type = ARG_INT
         },
         // 3
        {.name = "Batch size",
         .description = "Number of alignments per batch (default=num-alignments).",
         .short_arg = 'b',
         .long_arg = "batch-size",
         .required = false,
         .type = ARG_INT
         },
        // 4
        {.name = "Number of CPU threads",
         .description = "Number of CPU threads, each CPU thread creates two "
                        "streams to overlap compute and memory transfers. "
                        "(default=1)",
         .short_arg = 't',
         .long_arg = "cpu-threads",
         .required = false,
         .type = ARG_INT
         },
        // 5
        {.name = "Print CIGARS",
         .description = "Print CIGARS to stdout",
         .short_arg = 'p',
         .long_arg = "print-cigars",
         .required = false,
         .type = ARG_NO_VALUE
         }
    };

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        WF_FATAL("No available CUDA devices detected.")
    }

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using device \"%s\", with compute capability %d.%d\n",
           deviceProp.name, deviceProp.major, deviceProp.minor);


    options_t options = {options_arr, NUM_ARGUMENTS};
    bool success = parse_args(argc, argv, options);
    if (!success) {
        print_usage(options);
        exit(1);
    }

    char* seq_file = options.options[0].value.str_val;
    size_t num_alignments = options.options[1].value.int_val;
    size_t seq_len = options.options[2].value.int_val;

    size_t batch_size = num_alignments;
    if (options.options[3].parsed) {
        batch_size = options.options[3].value.int_val;
    }

    unsigned int num_threads = 1;
    if (options.options[4].parsed) {
        num_threads = options.options[4].value.int_val;
    }

    bool print_cigars = false;
    if (options.options[5].parsed) {
        print_cigars = true;
    }

    if (num_threads < 1)
        WF_FATAL("Minimum needed is 1 thread");

    if (batch_size > num_alignments)
        WF_FATAL("batch_size must be <= than the number of alignments.");

    pthread_t *threads = (pthread_t*)calloc(num_threads, sizeof(pthread_t));
    if (threads == NULL)
        WF_FATAL(NOMEM_ERR_STR);

    Parameters *params_array = (Parameters*)calloc(num_threads, sizeof(Parameters));

    SequenceReader reader = SequenceReader(seq_file, seq_len, num_alignments,
                                           batch_size);
    if (!reader.read_file()) {
        WF_FATAL("Could not read the sequences from file %s\n", seq_file);
    }

    CLOCK_INIT_NO_DEBUG()
    CLOCK_START_NO_DEBUG()
    
    for (unsigned int i=1; i<num_threads; i++) {
        Parameters *params = &params_array[i];
        params->seq_file = seq_file;
        params->seq_len = seq_len;
        params->num_alignments= num_alignments;
        params->batch_size = batch_size;
        params->tid = i;
        params->total_threads = num_threads;
        params->print_cigars = print_cigars;
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
    params->print_cigars = print_cigars;
    params->reader = &reader;
    worker(params);

    for (int i=1; i<num_threads; i++) {
        // tid is i+1 because thread 0 is the thread that launches the others
        if (pthread_join(threads[i], NULL))
            WF_FATAL("Can not join thread.");
    }

    CLOCK_STOP_NO_DEBUG("Aligned executed.")

    free(params_array);
    free(threads);

    return 0;
}
