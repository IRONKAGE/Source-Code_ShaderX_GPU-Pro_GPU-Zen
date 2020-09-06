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
#include "gpu_gauss.h"


static texture<float, 2, cudaReadModeElementType> texSRC1;
static texture<float4, 2, cudaReadModeElementType> texSRC4;

template<typename T> T texSRC(float x, float y);
template<> inline __device__ float texSRC(float x, float y) { return tex2D(texSRC1, x, y); }
template<> inline __device__ float4 texSRC(float x, float y) { return tex2D(texSRC4, x, y); }

static texture<float, 2, cudaReadModeElementType> texSIGMA;
struct texSIGMA_t { 
    inline __device__ float operator()(int ix, int iy) { return tex2D(texSIGMA, ix, iy); }
};


template<typename T> 
__global__ void imp_gauss_filter( gpu_plm2<T> dst, float sigma, float precision ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    float twoSigma2 = 2.0f * sigma * sigma;
    int halfWidth = int(ceilf( precision * sigma ));

    T sum = make_zero<T>();
    float norm = 0;
    for ( int i = -halfWidth; i <= halfWidth; ++i ) {
        for ( int j = -halfWidth; j <= halfWidth; ++j ) {
            float d = length(make_float2(i,j));
            float kernel = __expf( -d *d / twoSigma2 );
            T c = texSRC<T>(ix + i, iy + j);
            sum += kernel * c;
            norm += kernel;
        }
    }
    sum /=  norm;
    
    dst(ix, iy) = sum;
}


gpu_image<float> gpu_gauss_filter( const gpu_image<float>& src, float sigma, float precision ) {
    gpu_image<float> dst(src.size());
    bind(&texSRC1, src);
    imp_gauss_filter<float><<<dst.blocks(), dst.threads()>>>(dst, sigma, precision);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_gauss_filter( const gpu_image<float4>& src, float sigma, float precision ) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC4, src);
    imp_gauss_filter<float4><<<dst.blocks(), dst.threads()>>>(dst, sigma, precision);
    GPU_CHECK_ERROR();
    return dst;
}


template<typename T, int dx, int dy> 
__global__ void imp_gauss_filter_xy( gpu_plm2<T> dst, float sigma, float precision ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float twoSigma2 = 2.0f * sigma * sigma;
    int halfWidth = ceilf( precision * sigma );

    T sum = texSRC<T>(ix, iy);
    float norm = 1;
    for ( int i = 1; i <= halfWidth; ++i ) {
        float kernel = __expf( -i *i / twoSigma2 );
        sum += kernel * (texSRC<T>(ix + dx * i, iy + dy * i) + texSRC<T>(ix - dx * i, iy - dy * i));
        norm += 2 * kernel;
    }
    sum /=  norm;
    
    dst(ix, iy) = sum;
}


gpu_image<float> gpu_gauss_filter_xy( const gpu_image<float>& src, float sigma, float precision ) {
	if (sigma <= 0) return src;
	gpu_image<float> dst(src.size());
    gpu_image<float> tmp(src.size());
    bind(&texSRC1, src);
    imp_gauss_filter_xy<float,1,0><<<tmp.blocks(), tmp.threads()>>>(tmp, sigma, precision);
    GPU_CHECK_ERROR();
    bind(&texSRC1, tmp);
    imp_gauss_filter_xy<float,0,1><<<dst.blocks(), dst.threads()>>>(dst, sigma, precision);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_gauss_filter_xy( const gpu_image<float4>& src, float sigma, float precision ) {
	if (sigma <= 0) return src;
    gpu_image<float4> dst(src.size());
    gpu_image<float4> tmp(src.size());
    bind(&texSRC4, src);
    imp_gauss_filter_xy<float4,1,0><<<tmp.blocks(), tmp.threads()>>>(tmp, sigma, precision);
    GPU_CHECK_ERROR();
    bind(&texSRC4, tmp);
    imp_gauss_filter_xy<float4,0,1><<<dst.blocks(), tmp.threads()>>>(dst, sigma, precision);
    GPU_CHECK_ERROR();
    return dst;
}


// [0.216, 0.568, 0.216], sigma ~= 0.680
template<typename T> 
__global__ void imp_gauss_filter_3x3( gpu_plm2<T> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

     T sum =
        ( 0.046656f * texSRC<T>(ix-1, iy-1) +
          0.122688f * texSRC<T>(ix,   iy-1) +
          0.046656f * texSRC<T>(ix+1, iy-1) +
          0.122688f * texSRC<T>(ix-1, iy) +
          0.322624f * texSRC<T>(ix,   iy) +
          0.122688f * texSRC<T>(ix+1, iy) +
          0.046656f * texSRC<T>(ix-1, iy+1) +
          0.122688f * texSRC<T>(ix,   iy+1) +
          0.046656f * texSRC<T>(ix+1, iy+1)
        );

     dst(ix, iy) = sum;
}


gpu_image<float4> gpu_gauss_filter_3x3( const gpu_image<float4>& src) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC4, src);
    imp_gauss_filter_3x3<float4><<<dst.blocks(), dst.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}


// [0.03134, 0.24, 0.45732, 0.24, 0.03134], sigma ~= 0.867
template<typename T> 
__global__ void imp_gauss_filter_5x5( gpu_plm2<T> dst ) {
    const float kernel[5][5] = {
        { 0.0009821956f, 0.0075216f, 0.0143324088f, 0.0075216f, 0.0009821956 },
        { 0.0075216f,    0.0576f,    0.1097568f,    0.0576f,    0.0075216 },
        { 0.0143324088f, 0.1097568f, 0.2091415824f, 0.1097568f, 0.0143324088 },
        { 0.0075216f,    0.0576f,    0.1097568f,    0.0576f,    0.0075216 },
        { 0.0009821956f, 0.0075216f, 0.0143324088f, 0.0075216f, 0.0009821956 }
    };

    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    T sum = make_zero<T>();
    for ( int j = 0; j < 5; ++j ) {
        for ( int i = 0; i < 5; ++i ) {
            T c = texSRC<T>(ix + i -2, iy + j - 2);
            sum += kernel[j][i] * c;
        }
    }
    
    dst(ix, iy) = sum;
}


gpu_image<float4> gpu_gauss_filter_5x5( const gpu_image<float4>& src) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC4, src);
    imp_gauss_filter_5x5<float4><<<dst.blocks(), dst.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}
