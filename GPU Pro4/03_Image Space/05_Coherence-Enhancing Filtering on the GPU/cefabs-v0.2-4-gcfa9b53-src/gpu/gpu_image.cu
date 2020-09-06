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
#include "gpu_image.h"


template <typename T>
__global__ void imp_set( gpu_plm2<T> dst, T value) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    dst(ix, iy) = value;
}


gpu_image<float> gpu_set( float value, unsigned w, unsigned h ) {
    gpu_image<float> dst(w,h);
    imp_set<float><<<dst.blocks(), dst.threads()>>>(  dst, value );
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_set( float4 value, unsigned w, unsigned h ) {
    gpu_image<float4> dst(w,h);
    imp_set<float4><<<dst.blocks(), dst.threads()>>>(  dst, value );
    GPU_CHECK_ERROR();
    return dst;
}


template <typename T>
__global__ void imp_add( gpu_plm2<T> dst, const gpu_plm2<T> src0, const gpu_plm2<T> src1) {
    const unsigned ix = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    dst(ix,iy) = src0(ix,iy) + src1(ix,iy);
}                       


gpu_image<float> gpu_add( const gpu_image<float>& src0, const gpu_image<float>& src1 ) {
    assert(src0.size() == src1.size());
    gpu_image<float> dst(src0.size());
    imp_add<float><<<dst.blocks(), dst.threads()>>>(dst, src0, src1);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_add( const gpu_image<float4>& src0, const gpu_image<float4>& src1 ) {
    assert(src0.size() == src1.size());
    gpu_image<float4> dst(src0.size());
    imp_add<float4><<<dst.blocks(), dst.threads()>>>(dst, src0, src1);
    GPU_CHECK_ERROR();
    return dst;
}


template <typename T>
__global__ void imp_mul( const gpu_plm2<T> src, gpu_plm2<T> dst, float value) {
    const unsigned ix = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    T c = src(ix,iy);
    dst(ix,iy) = c * value;
}                       


gpu_image<float> gpu_mul( const gpu_image<float>& src, float value ) {
    gpu_image<float> dst(src.size());
    imp_mul<float><<<dst.blocks(), dst.threads()>>>(src, dst, value);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_mul( const gpu_image<float4>& src, float value ) {
    gpu_image<float4> dst(src.size());
    imp_mul<float4><<<dst.blocks(), dst.threads()>>>(src, dst, value);
    GPU_CHECK_ERROR();
    return dst;
}

