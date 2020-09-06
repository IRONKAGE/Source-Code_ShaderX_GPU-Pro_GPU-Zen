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

#include "cpu_image.h"
#include "gpu_error.h"
#include "basic_image.h"
#include "gpu_plm2.h"
#include "gpu_cache.h"


struct gpu_allocator {
    void alloc(void **ptr, unsigned *pitch, unsigned w, unsigned h) {
        gpu_cache_alloc(ptr, pitch, w, h);
    }
    void free(void *ptr, unsigned pitch, unsigned w, unsigned h) {
        gpu_cache_free(ptr, pitch, w, h);
    }
};


typedef basic_image_data<gpu_allocator> gpu_image_data;


template <typename T> class gpu_image : public basic_image<T,gpu_allocator> {
public:
    gpu_image() : basic_image<T,gpu_allocator>() { }
    gpu_image(unsigned w, unsigned h) : basic_image<T,gpu_allocator>(w, h) { }
    gpu_image(uint2 size) : basic_image<T,gpu_allocator>(size) { }
    gpu_image(gpu_image_data *data) : basic_image<T,gpu_allocator>(data) { }

    gpu_image(const T *src, size_t src_pitch, unsigned w, unsigned h) : basic_image<T,gpu_allocator>(w, h) {
        copy(this, src, src_pitch);
    }

    gpu_image(const T *src, unsigned w, unsigned h) : basic_image<T,gpu_allocator>(w, h) {
        copy(this, src, w * sizeof(T));
    }

    gpu_image(const cpu_image<T>& src) :  basic_image<T,gpu_allocator>(src.w(), src.h()){
        copy(this, src.ptr(), src.pitch());
    }

    gpu_image(const gpu_image& img) : basic_image<T,gpu_allocator>(img) { }

    const gpu_image& operator=(const gpu_image& img) {
        basic_image<T,gpu_allocator>::operator=(img);
        return *this;
    }

    cpu_image<T> cpu() const {
        if (!this->is_valid()) return cpu_image<T>();
        cpu_image<T> dst(this->w(), this->h());
        copy(dst.ptr(), dst.pitch(), this);
        return dst;
    }

    dim3 threads() const {
        return dim3(8, 8);
    }

    dim3 blocks(dim3 threads = dim3(8, 8)) const {
        return dim3( (int)ceil((float)this->w() / threads.x), (int)ceil((float)this->h() / threads.y) );
    }

    operator gpu_plm2<T>() {
        if (!this->is_valid()) return gpu_plm2<T>();
        return gpu_plm2<T>((T*)this->ptr(), this->pitch(), this->w(), this->h());
    }

    operator const gpu_plm2<T>() const {
        if (!this->is_valid()) return gpu_plm2<T>();
        return gpu_plm2<T>((T*)this->ptr(), this->pitch(), this->w(), this->h());
    }

    void zero() {
        GPU_SAFE_CALL(cudaMemset2D(this->ptr(), this->pitch(), 0, this->w()*sizeof(T), this->h()));
    }
};


template <typename T> 
void copy(gpu_image<T>* dst, const void *src, size_t src_pitch) {
    GPU_SAFE_CALL(cudaMemcpy2D(dst->ptr(), dst->pitch(), src, src_pitch, dst->w()*sizeof(T), 
                               dst->h(), cudaMemcpyHostToDevice));
}

template <typename T> 
void copy(void *dst, size_t dst_pitch, const gpu_image<T>* src) {
    GPU_SAFE_CALL(cudaMemcpy2D(dst, dst_pitch, src->ptr(), src->pitch(), src->w()*sizeof(T), 
                               src->h(), cudaMemcpyDeviceToHost));
}


#ifdef __CUDACC__

template <typename T> 
void bind(const texture<T,2>* tex, const gpu_image<T>& img) {
    GPU_SAFE_CALL(cudaBindTexture2D(0, *tex, img.ptr(), img.w(), img.h(), img.pitch()));
}

#endif

gpu_image<float> gpu_set( float value, unsigned w, unsigned h );
gpu_image<float4> gpu_set( float4 value, unsigned w, unsigned h );
gpu_image<float> gpu_add( const gpu_image<float>& src0, const gpu_image<float>& src1 );
gpu_image<float4> gpu_add( const gpu_image<float4>& src0, const gpu_image<float4>& src1 );
gpu_image<float> gpu_mul( const gpu_image<float>& src, float value );
gpu_image<float4> gpu_mul( const gpu_image<float4>& src, float value );
