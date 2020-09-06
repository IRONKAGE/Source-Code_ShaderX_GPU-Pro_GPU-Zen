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
#pragma once

template <typename T>
struct gpu_plm2 {
    T *ptr;
    unsigned stride;
    unsigned w;
    unsigned h;

    __host__ gpu_plm2() {
        ptr = 0;
        stride = w = h = 0;
    }

    __host__ gpu_plm2(T *ptr, unsigned pitch, unsigned w, unsigned h) {
        this->ptr = ptr;
        this->stride = pitch / sizeof(T);
        this->w = w;
        this->h = h;
    }

    #ifdef __CUDACC__

    inline __device__ T& operator()(int x, int y) { 
        return ptr[y * stride + x];
    }

    inline __device__ const T& operator()(int x, int y) const { 
        return ptr[y * stride + x];
    }

    #endif
};
