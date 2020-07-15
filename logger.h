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

#ifndef LOGGER_H
#define LOGGER_H

#ifdef DEBUG_MODE
#define DEBUG(...) {\
    char tmp[512];\
    snprintf(tmp, 512, __VA_ARGS__); \
    fprintf(stderr, "DEBUG: %s (%s:%d)\n", tmp, __FILE__, __LINE__); \
    }
#define DEBUG_GREEN(...) {\
    char tmp[512];\
    snprintf(tmp, 512, __VA_ARGS__); \
    fprintf(stderr, "\u001b[32mDEBUG: %s (%s:%d)\u001b[0m\n", tmp, __FILE__, __LINE__); \
    }
#define DEBUG_RED(...) {\
    char tmp[512];\
    snprintf(tmp, 512, __VA_ARGS__); \
    fprintf(stderr, "\u001b[31mDEBUG: %s (%s:%d)\u001b[0m\n", tmp, __FILE__, __LINE__); \
    }
#define CLOCK_INIT() struct timespec now, tmstart; double seconds;
#define CLOCK_START() clock_gettime(CLOCK_REALTIME, &tmstart);
#define CLOCK_STOP(text) \
    clock_gettime(CLOCK_REALTIME, &now); \
    seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9)); \
    DEBUG("%s Wall time %fs", text, seconds);
#else
#define DEBUG(fmt, ...)
#define CLOCK_INIT()
#define CLOCK_START()
#define CLOCK_STOP(text)
#endif

#define NOMEM_ERR_STR "Could not allocate memory.\n"

#define WF_ERROR(...) fprintf(stderr, __VA_ARGS__)

// TODO: Handle error before exiting? (free gpu memory?)
#define WF_FATAL(...) { \
    WF_ERROR(__VA_ARGS__); fflush(stdout); fflush(stderr); exit(1); \
    }

#endif // Header guard
