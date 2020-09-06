//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2012 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
#include "gpu_error.h"
#ifdef WIN32
#include <sstream>
#include <windows.h>
#else
#include <cstdio>
#include <cstdlib>
#endif


void gpu_error_msg(cudaError_t err, const char *file, size_t line) {
#ifdef WIN32
    if (!IsDebuggerPresent()) {
        std::ostringstream oss;
        oss << cudaGetErrorString(err) << "\n"
            << file << "(" << line << ")";
        MessageBoxA(NULL, oss.str().c_str(), "CUDA Error", MB_OK | MB_ICONERROR);
    } else {
        OutputDebugStringA("CUDA Error: ");
        OutputDebugStringA(cudaGetErrorString(err));
        OutputDebugStringA("\n");
        DebugBreak();
    }
#else
    fprintf(stderr, "%s(%d): CUDA Error\n", file, (int)line);
    fprintf(stderr, "%s\n", cudaGetErrorString(err));
#endif
    exit(1);
}
