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

#include <cstring>
#include <cassert>
#include "gpu_math.h"
#include "basic_image.h"
#include <cstdlib>


class cpu_allocator {
public:
    cpu_allocator(bool locked=false) 
        : m_locked(locked) 
    { }

    void alloc(void **ptr, unsigned *pitch, unsigned w, unsigned h) {
        // TODO: support for host locked memory
        *pitch = (4 * w + 3) / 4;
        *ptr = ::malloc(*pitch * h);
        assert(*ptr);
        //*ptr = _aligned_malloc(*pitch * h, 16);
    }
    void free(void *ptr, unsigned pitch, unsigned w, unsigned h) {
        // TODO: support for host locked memory
        ::free(ptr);
        //_aligned_free(ptr);
    }

    bool is_locked() const { return m_locked; }

private:
    bool m_locked;
};

typedef basic_image_data<cpu_allocator> cpu_image_data;


template <typename T> class cpu_image : public basic_image<T,cpu_allocator> {
public:
    cpu_image() : basic_image<T,cpu_allocator>() { }
    cpu_image(unsigned w, unsigned h, bool locked=false) : basic_image<T,cpu_allocator>(w, h, cpu_allocator(locked)) { }
    cpu_image(uint2 size) : basic_image<T,cpu_allocator>(size) { }
    cpu_image(cpu_image_data *data) : basic_image<T,cpu_allocator>(data) { }

    cpu_image(const T *src, size_t src_pitch, unsigned w, unsigned h) : basic_image<T,cpu_allocator>(w, h) {
        copy(this, src, src_pitch);
    }

    cpu_image(const T *src, unsigned w, unsigned h) : basic_image<T,cpu_allocator>(w, h) {
        copy(this, src, w * sizeof(T));
    }

    cpu_image(const cpu_image& img) : basic_image<T,cpu_allocator>(img) { }

    const cpu_image& operator=(const cpu_image& img) {
        basic_image<T,cpu_allocator>::operator=(img);
        return *this;
    }

    T* operator[](int y) { 
        return reinterpret_cast<T*>(this->ptr8u() + y * this->pitch());
    }

    const T* operator[](int y) const { 
        return reinterpret_cast<const T*>(this->ptr8u() + y * this->pitch());
    }

    T& operator()(int x, int y) { 
        if (x < 0) x = 0; else if (x >= (int)this->w()) x = this->w() - 1;
        if (y < 0) y = 0; else if (y >= (int)this->h()) y = this->h() - 1;
        return (*this)[y][x];
    }

    const T& operator()(int x, int y) const { 
        if (x < 0) x = 0; else if (x >= (int)this->w()) x = this->w() - 1;
        if (y < 0) y = 0; else if (y >= (int)this->h()) y = this->h() - 1;
        return (*this)[y][x];
    }

    T& operator()(float x, float y) { 
        return operator()((int)x, (int)y);
    }

    const T& operator()(float x, float y) const { 
        return operator()((int)x, (int)y);
    }

    T& operator()(double x, double y) { 
        return operator()((int)x, (int)y);
    }

    const T& operator()(double x, double y) const { 
        return operator()((int)x, (int)y);
    }

    // TODO: optimize
    T sample_linear(float x, float y) const {
        x -= 0.5f;
        y -= 0.5f;

        int x0 = (int)x;
        int y0 = (int)y;
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float fx = x - floor(x);
        float fy = y - floor(y);

        T c0 = operator()(x0, y0);
        T c1 = operator()(x1, y0);
        T c2 = operator()(x0, y1);
        T c3 = operator()(x1, y1);

        return (1 - fy) * ((1 - fx) * c0 + fx * c1) + fy * ((1 - fx) * c2 + fx * c3);
    }
};


template <typename T> 
void copy(cpu_image<T>* dst, const void *src, size_t src_pitch) {
    cudaMemcpy2D(dst->ptr(), dst->pitch(), src, src_pitch, dst->w()*sizeof(T), dst->h(), cudaMemcpyHostToHost);
}

template <typename T> 
void copy(void *dst, size_t dst_pitch, const cpu_image<T>* src) {
    cudaMemcpy2D(dst, dst_pitch, src->ptr(), src->pitch(), src->w()*sizeof(T), src->h(), cudaMemcpyHostToHost);
}
