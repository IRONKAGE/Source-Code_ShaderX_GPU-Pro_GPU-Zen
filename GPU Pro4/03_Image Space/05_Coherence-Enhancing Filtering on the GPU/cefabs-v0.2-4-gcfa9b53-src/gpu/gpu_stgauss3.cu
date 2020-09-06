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
#include "gpu_stgauss3.h"
#include "gpu_st.h"


static texture<float4, 2> s_texSRC;
static texture<float4, 2> s_texST;


struct SRC_sampler {
    SRC_sampler( const gpu_image<float4>& src, cudaTextureFilterMode filter_mode ) {
        s_texSRC.filterMode = filter_mode;
        GPU_SAFE_CALL(cudaBindTexture2D(0, s_texSRC, src.ptr(), src.w(), src.h(), src.pitch()));
    }

    ~SRC_sampler() {
        s_texSRC.filterMode = cudaFilterModePoint;
        cudaUnbindTexture(s_texSRC);
    }

    inline __device__ float4 operator()(float x, float y) const {
        return tex2D(s_texSRC, x, y);
    }
};


struct ST_sampler {
    ST_sampler( const gpu_image<float4>& src, cudaTextureFilterMode filter_mode ) {
        s_texST.filterMode = filter_mode;
        GPU_SAFE_CALL(cudaBindTexture2D(0, s_texST, src.ptr(), src.w(), src.h(), src.pitch()));
    }

    ~ST_sampler() {
        s_texST.filterMode = cudaFilterModePoint;
        cudaUnbindTexture(s_texST);
    }

    inline __device__ float4 operator()(float x, float y) const {
        return tex2D(s_texST, x, y);
    }
};


struct stgauss3_filter {
     __device__ stgauss3_filter( float sigma ) {
        radius_ = 2 * sigma;
        twoSigma2_ = 2 * sigma * sigma;
        c_ = make_float3(0);
        w_ = 0;
    }

    __device__ float radius() const {
        return radius_;
    }

    __device__ void operator()(float u, float2 p) {
        float k = __expf(-u * u / twoSigma2_);
        c_ += k * make_float3(tex2D(s_texSRC, p.x, p.y));
        w_ += k;
    }

    float radius_;
    float twoSigma2_;
    float3 c_;
    float w_;
};


template<int order, typename SRC, typename ST>
__global__ void imp_stgauss3_filter( gpu_plm2<float4> dst, SRC src, ST st, float sigma,
                                     float step_size, bool adaptive )
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
    stgauss3_filter f(sigma);
    if (order == 2) {
        st_integrate_rk2(p0, st, f, dst.w, dst.h, step_size, adaptive);
    } else {
        st_integrate_euler(p0, st, f, dst.w, dst.h, step_size, adaptive);
    }
    dst(ix, iy) = make_float4(f.c_ / f.w_, 1);
}


gpu_image<float4> gpu_stgauss3_filter( const gpu_image<float4>& src, const gpu_image<float4>& st,
                                       float sigma, bool src_linear, bool st_linear,
                                       int order, float step_size, bool adaptive )
{
    assert(src.size() == st.size());
    if (sigma <= 0) return src;
    gpu_image<float4> dst(src.size());

    SRC_sampler src_sampler(src, src_linear? cudaFilterModeLinear : cudaFilterModePoint);
    ST_sampler st_sampler(st, st_linear? cudaFilterModeLinear : cudaFilterModePoint);

    if (order == 2)
        imp_stgauss3_filter<2><<<dst.blocks(), dst.threads()>>>(dst, src_sampler, st_sampler, sigma, step_size, adaptive);
    else
        imp_stgauss3_filter<1><<<dst.blocks(), dst.threads()>>>(dst, src_sampler, st_sampler, sigma, step_size, adaptive);

    GPU_CHECK_ERROR();
    return dst;
}
