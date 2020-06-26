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

const char *USAGE_STR = "Usage:\n"
                        "WFA_edit_gpu <file> <sequence_length> <num_of_sequences> "
                        "<batch_size=num_of_sequences>\n";

int main (int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        WF_FATAL(USAGE_STR);
    }

    char* seq_file = argv[1];
    size_t seq_len = atoi(argv[2]);
    size_t num_sequences = atoi(argv[3]);
    size_t batch_size = (argc == 5) ? atoi(argv[4]) : num_sequences;

    SequenceReader reader = SequenceReader(seq_file, seq_len, num_sequences);
    if (!reader.read_sequences()) {
        WF_FATAL("Could not read the sequences from file %s\n", argv[1]);
    }

    Sequences seqs = Sequences(reader.sequences, num_sequences, seq_len, batch_size);
    seqs.GPU_memory_init();
    seqs.GPU_launch_wavefront_distance();
    seqs.GPU_memory_free();
    return 0;
}
