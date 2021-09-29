/*
 * Copyright (c) 2021 Quim Aguado
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

#ifndef ARG_HANDLER_H
#define ARG_HANDLER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ARG_INT,
    ARG_STR,
    ARG_CHAR,
    ARG_FLOAT,
    ARG_BOOL,
    ARG_NO_VALUE
} arg_type_t;

typedef union {
    int64_t int_val;
    char* str_val;
    char char_val;
    double float_val;
    bool bool_val;
    bool set;
} arg_value_t;

typedef struct {
    const char* name;
    const char* description;
    char short_arg; // -a without the "-"
    const char* long_arg; // --arg without the "--"
    bool required;
    bool parsed;
    arg_type_t type;
    arg_value_t value;
} option_t;

typedef struct {
    option_t* options;
    int len;
} options_t;

bool parse_args (const int argc, char** argv, options_t options);
void print_usage (const options_t options);

#ifdef __cplusplus
}
#endif // cplusplus

#endif // Header guard
