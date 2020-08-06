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

#ifndef WAVEFRONT_STRUCTURES_H
#define WAVEFRONT_STRUCTURES_H

#include <stdint.h>

#define SEQ_TYPE char

typedef int16_t ewf_offset_t;
typedef char edit_cigar_t;

#define OFFSETS_TOTAL_ELEMENTS(max_d) (max_d * max_d + (max_d * 2 + 1))
#define WF_ELEMENTS(d) (d * 2 + 1)
#define OFFSETS_PTR(offsets_mem, d) (offsets_mem + ((d)*(d)) + (d))

struct edit_wavefronts_t {
    uint32_t d;
    ewf_offset_t* offsets;
};

#define TEXT_PTR(idx, base_ptr, max_seq_len) ((SEQ_TYPE*)(base_ptr + idx * 2 * max_seq_len))
#define PATTERN_PTR(idx, base_ptr, max_seq_len) ((SEQ_TYPE*)(base_ptr + (idx * 2 + 1) * max_seq_len))

struct WF_element {
    // With the idx, the max_seq_length and the sequences memory base pointer,
    // is possible to calculate the text/pattern address
    int alignment_idx;
    size_t tlen;
    size_t plen;
};

//backtrace->words[word] = (prev_backtrace->words[word] << 2) | op;
// Assume 128 bit backtrace, composed of 2 64bit words
//#define WRITE_BT_OP(backtrace, prev_backtrace, d, op) \
//    int word = d / 32; \
//    backtrace->words[word] = prev_backtrace->words[word] | (((uint64_t)op) << ((d % 32) * 2));

// 128 bit backtrace data, max_distance = 64
struct WF_backtrace_t {
    uint64_t words[2];
    int distance;
};

#define CURR_MAX_DISTANCE 64

// Backtrace operations encoded in 2 bits, using more than 2 bits will break the
// code, so don't do it.
typedef enum {
    OP_NOOP, // Use for delimitation
    OP_DEL = 3,  // k + 1, "going up"
    OP_SUB = 2,  // k
    OP_INS = 1   // k - 1, "going down"
} WF_backtrace_op_t;

#endif
