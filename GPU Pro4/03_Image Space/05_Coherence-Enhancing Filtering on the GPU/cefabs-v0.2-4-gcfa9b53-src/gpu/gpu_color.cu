//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2011 Computer Graphics Systems Group at the
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
#include "gpu_color.h"


__device__ float srgb2linear( float x ) {
    return ( x > 0.04045f ) ? __powf( ( x + 0.055f ) / 1.055f, 2.4f ) : x / 12.92f;
}


__device__ float linear2srgb( float x ) {
    return ( x > 0.0031308f ) ? (( 1.055f * __powf( x, ( 1.0f / 2.4f ))) - 0.055f ) : 12.92f * x;
}


__device__ float3 rgb2xyz( float4 c ) {
    float b = ( c.x > 0.04045f ) ? __powf( ( c.x + 0.055f ) / 1.055f, 2.4f ) : c.x / 12.92f;
    float g = ( c.y > 0.04045f ) ? __powf( ( c.y + 0.055f ) / 1.055f, 2.4f ) : c.y / 12.92f;
    float r = ( c.z > 0.04045f ) ? __powf( ( c.z + 0.055f ) / 1.055f, 2.4f ) : c.z / 12.92f;
    return make_float3(
        100 * (0.4124f * r + 0.3576f * g + 0.1805f * b),
        100 * (0.2126f * r + 0.7152f * g + 0.0722f * b),
        100 * (0.0193f * r + 0.1192f * g + 0.9505f * b)
    );
}


__device__ float4 xyz2rgb( float x, float y, float z ) {
    float r = ( 3.2406f * x - 1.5372f * y - 0.4986f * z ) / 100.0f;
    float g = (-0.9689f * x + 1.8758f * y + 0.0415f * z ) / 100.0f;
    float b = ( 0.0557f * x - 0.2040f * y + 1.0570f * z ) / 100.0f;
    return make_float4(
        ( b > 0.0031308f ) ? (( 1.055f * __powf( b, ( 1.0f / 2.4f ))) - 0.055f ) : 12.92f * b,
        ( g > 0.0031308f ) ? (( 1.055f * __powf( g, ( 1.0f / 2.4f ))) - 0.055f ) : 12.92f * g,
        ( r > 0.0031308f ) ? (( 1.055f * __powf( r, ( 1.0f / 2.4f ))) - 0.055f ) : 12.92f * r,
        1
    );
}


__global__ void imp_srgb2linear( const gpu_plm2<float> src, gpu_plm2<float> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;
    dst(ix, iy) = srgb2linear(src(ix, iy));
}


gpu_image<float> gpu_srgb2linear( const gpu_image<float>& src) {
    gpu_image<float> dst(src.size());
    imp_srgb2linear<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_linear2srgb( const gpu_plm2<float> src, gpu_plm2<float> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;
    dst(ix, iy) = linear2srgb(src(ix, iy));
}


gpu_image<float> gpu_linear2srgb( const gpu_image<float>& src) {
    gpu_image<float> dst(src.size());
    imp_linear2srgb<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_rgb2lab( const gpu_plm2<float4> src, gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float3 c = rgb2xyz( src(ix, iy) );
    c.x /= 95.047f;
    c.y /= 100.0f;
    c.z /= 108.883f;

    float x = ( c.x > 0.008856f ) ? pow( c.x, 1.0f / 3.0f ) : ( 7.787f * c.x ) + ( 16.0f / 116.0f );
    float y = ( c.y > 0.008856f ) ? pow( c.y, 1.0f / 3.0f ) : ( 7.787f * c.y ) + ( 16.0f / 116.0f );
    float z = ( c.z > 0.008856f ) ? pow( c.z, 1.0f / 3.0f ) : ( 7.787f * c.z ) + ( 16.0f / 116.0f );

    dst(ix, iy) = make_float4(
        ( 116 * y ) - 16, 
        500 * ( x - y ), 
        200 * ( y - z ),
        1
    );
}


gpu_image<float4> gpu_rgb2lab( const gpu_image<float4>& src) {
    gpu_image<float4> dst(src.size());
    imp_rgb2lab<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_lab2rgb( const gpu_plm2<float4> src, gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float4 c = src(ix, iy);
    float fy = ( c.x + 16.0f ) / 116.0f;
    float fx = c.y / 500.0f + fy;
    float fz = fy - c.z / 200.0f;
    dst(ix, iy) = xyz2rgb(
         95.047f * (( fx > 0.206897f ) ? fx * fx * fx : ( fx - 16.0f / 116.0f ) / 7.787f),
        100.000f * (( fy > 0.206897f ) ? fy * fy * fy : ( fy - 16.0f / 116.0f ) / 7.787f),
        108.883f * (( fz > 0.206897f ) ? fz * fz * fz : ( fz - 16.0f / 116.0f ) / 7.787f)
    );
}


gpu_image<float4> gpu_lab2rgb( const gpu_image<float4>& src) {
    gpu_image<float4> dst(src.size());
    imp_lab2rgb<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_gray2rgb( const gpu_plm2<float> src, gpu_plm2<float4> dst, bool saturate ) 
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float c = src(ix, iy);
    if (saturate) c = __saturatef(c);
    dst(ix, iy) = make_float4(c, c, c, 1);
}


gpu_image<float4> gpu_gray2rgb( const gpu_image<float>& src, bool saturate ) {
    gpu_image<float4> dst(src.size());
    imp_gray2rgb<<<dst.blocks(), dst.threads()>>>(src, dst, saturate);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_rgb2gray( const gpu_plm2<float4> src, gpu_plm2<float> dst ) 
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float4 c = src(ix, iy);
    dst(ix, iy) = 0.299f * __saturatef(c.z) + 
                  0.587f * __saturatef(c.y) + 
                  0.114f * __saturatef(c.x);
}


gpu_image<float> gpu_rgb2gray( const gpu_image<float4>& src ) {
    gpu_image<float> dst(src.size());
    imp_rgb2gray<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_swap_rgba( gpu_plm2<float4> dst, const gpu_plm2<float4> src ) 
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float4 c = src(ix, iy);
    dst(ix, iy) = make_float4(c.z, c.y, c.x, c.w);
}


gpu_image<float4> gpu_swap_rgba( const gpu_image<float4>& src ) {
    gpu_image<float4> dst(src.size());
    imp_swap_rgba<<<dst.blocks(), dst.threads()>>>(dst, src);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_colorize_sign( gpu_plm2<float4> dst, const gpu_plm2<float> src, float scale ) 
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float c = src(ix, iy);
    float H = clamp(c * scale, -1.0f, 1.0f);
    dst(ix, iy) = make_float4( 0, (H > 0)? H : 0, (H < 0)? -H : 0, 1);
}


gpu_image<float4> gpu_colorize_sign( const gpu_image<float>& src, float scale ) {
    gpu_image<float4> dst(src.size());
    imp_colorize_sign<<<dst.blocks(), dst.threads()>>>(dst, src, scale);
    GPU_CHECK_ERROR();
    return dst;
}
