# WFE-GPU

WFE-GPU is an implementation of the [wavefront alignment algorithm](https://github.com/smarco/WFA)
(WFA) to accelerate edit distance on GPU devices, producing the full alignment
CIGAR. It uses optimization techniques to dramatically reduce the amount of
memory needed by the WFA algorithm, being able to fit more data in the fast
memories of the GPU. Additionally, compute and memory transfers are fully
asynchronous and overlapped.

## Install

Make sure you have installed an updated [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
and `nvcc` is on your PATH.

To compile with alignment correctness checks and debug messages:
```
$ make debug
```

By default, CUDA compute capability 7.0 is used, it can be changed to compile
for other architectures, this is an example of compiling the binary for compute
capability 8.0:

```
$ make SM=80 COMPUTE=80 debug
```

To compile with the most performant version (without alignment correctness
checks):
```
$ make all
```

## Usage 

Executing the binary without arguments prints the usage options.

```
$ ./wfe.aligner
Options:
        -f, --file                          (string, required) Sequences file: File containing the sequences to align.
        -n, --num-alignments                (int, required) Number of alignments: Number of alignments to read from the file (default=all alignments)
        -l, --seq-len                       (int, required) Sequence length: Maximum sequence length.
        -b, --batch-size                    (int) Batch size: Number of alignments per batch (default=num-alignments).
        -t, --cpu-threads                   (int) Number of CPU threads: Number of CPU threads, each CPU thread creates two streams to overlap compute and memory transfers. (default=1)
```

The program takes as an input file datasets containing pairs of sequences, where
patterns start with `>` and texts start with `>`.

```
>TGTGAAGTAATGGACGTTCTATTGGTTAAGAAATGCACCAGCTACAGCAAACTATGAGTCATCCTTTTCCATGTTAAGCCTGGTTCCTAAACACTTCGTGAAGGACGAAACTTATGCACGCGTCTGCCCAACAGAAATCCTTCGTAACCG
<TGTAAAGTAATGGACGTTCTATTGGTTAAGAAATGCACCAGCTACAGCCAAACTATGAGTCATCCTTTTCCATGTTAAGCCTGGTTCCTAAACACTTCGTGAAGGACGAAACTTATGCACGCGTCTGCCCAACAGAAATCCTTCGTAACCG
>ACGGGCGTGCATCACAACCCGTGATGATCGCCATAGAGCGAGGGGTGGATATGGAGACCGTGTTGACGGTCTCACATATATTTGGTCTAGCACCTTCCGACATGACTTCGTCCTAATCTTACTCGTCAAAACAAAACAATGACAAGATAA
<ACGGGCGTGCATCACAACCCGGATGATCGCCATAGAGCCGAGGGGTGGATATGGAGACCGTGTTGACGGTCTCACATATATTTGGTCTAGCACCTTCCGACATGACTTCGATCCTAATCTTACTCGTCAAAACAAAACAATGACAAGATAA
>ATACCCCCGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGATGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG
<ATCCCACGTCTTATCATACGACCCTAATGCACGCGTTAGGGCGGCTTAAATCCCTCCTATCCCTGATGCCATTTGATGTGAAACTCGTGGCTAAGAAACGCCCAACTGGTCGTCTTTGTCCACCCTGGAAACGCGGGCACCCTCTTAG
...
```
