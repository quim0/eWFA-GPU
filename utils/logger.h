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

#define CLOCK_INIT_NO_DEBUG() struct timespec now, tmstart; double seconds;
#define CLOCK_START_NO_DEBUG() clock_gettime(CLOCK_REALTIME, &tmstart);
#define CLOCK_STOP_NO_DEBUG(text) \
    clock_gettime(CLOCK_REALTIME, &now); \
    seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9)); \
    printf("%s Wall time %fs\n", text, seconds);

#define NOMEM_ERR_STR "Could not allocate memory.\n"

#define WF_ERROR(...) fprintf(stderr, __VA_ARGS__)

// TODO: Handle error before exiting? (free gpu memory?)
#define WF_FATAL(...) { \
    WF_ERROR(__VA_ARGS__); fflush(stdout); fflush(stderr); exit(1); \
    }

// https://developer.nvidia.com/blog/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

#endif // Header guard
