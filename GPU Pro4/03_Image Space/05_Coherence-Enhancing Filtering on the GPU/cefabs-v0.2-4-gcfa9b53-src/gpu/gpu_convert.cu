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
#include "gpu_convert.h"


__global__ void imp_8u_to_32f( const gpu_plm2<unsigned char> src, gpu_plm2<float> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    unsigned char c = src(ix, iy);
    dst(ix, iy) = c / 255.0f;
}                       


__global__ void imp_8u_to_32f( const gpu_plm2<uchar4> src, gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    uchar4 c = src(ix, iy);
    dst(ix, iy) = make_float4(c.x / 255.0f, c.y / 255.0f, c.z / 255.0f, c.w / 255.0f);
}                       


__global__ void imp_32f_to_8u( const gpu_plm2<float> src, gpu_plm2<unsigned char> dst) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float c = clamp(src(ix, iy), 0.0f, 1.0f);
    dst(ix, iy) = (unsigned char)(255.0f *c);
}                       


__global__ void imp_32f_to_8u( const gpu_plm2<float4> src, gpu_plm2<uchar4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float4 c = clamp(src(ix, iy), 0, 1);
    dst(ix, iy) = make_uchar4((int)(255.0f *c.x), (int)(255.0f *c.y), (int)(255.0f *c.z), (int)(255.0f *c.w));
}                       


gpu_image<float> gpu_8u_to_32f( const gpu_image<unsigned char>& src ) {
    gpu_image<float> dst(src.size());
    imp_8u_to_32f<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_8u_to_32f( const gpu_image<uchar4>& src ) {
    gpu_image<float4> dst(src.size());
    GPU_CHECK_ERROR();
    imp_8u_to_32f<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<unsigned char> gpu_32f_to_8u( const gpu_image<float>& src ) {
    gpu_image<unsigned char> dst(src.size());
    imp_32f_to_8u<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<uchar4> gpu_32f_to_8u( const gpu_image<float4>& src ) {
    gpu_image<uchar4> dst(src.size());
    imp_32f_to_8u<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}
