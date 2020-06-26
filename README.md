# WFA.distance.gpu

PoC of WFA distance algorithm on GPU

# Install

You just need `nvcc` on your PATH. To install the debug version (with verbose output).
```
$ make debug
```

To install the "production" version
```
$ make
```

# Run

Run the binary without arguments to see the options

```
$ ./wfa.edit.distance.gpu
Usage:
WFA_edit_gpu <file> <sequence_length> <num_of_sequences> <batch_size=num_of_sequences>
```

* file: File where the sequences are stored
* sequence_length: length of the sequences (in this version all of them must have the same length)
* num_of_sequences: Number of alignments (sequences pairs). Usually half the number of lines of the sequences file.
* batch_size (optional): Size of the batch if you want the alignments to be processed in batches.
