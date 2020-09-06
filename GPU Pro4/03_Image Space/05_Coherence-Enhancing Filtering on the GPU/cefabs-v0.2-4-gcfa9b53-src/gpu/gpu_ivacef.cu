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
#include "gpu_ivacef.h"
#include "gpu_color.h"
#include "gpu_gauss.h"
#include "gpu_st.h"
#include "gpu_stgauss3.h"


static texture<float4, 2, cudaReadModeElementType> texSRC;
static texture<float4, 2, cudaReadModeElementType> texST;
static texture<float, 2, cudaReadModeElementType> texL;


__global__ void cef_scharr( gpu_plm2<float4> dst ) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) return;

    const float b1 = 46.84f / 256;
    const float b0 = 1 - 2 * b1;

    float3 g1 = 0.5f * (
          -b1 * make_float3(tex2D(texSRC, ix-1, iy-1)) +
          -b0 * make_float3(tex2D(texSRC, ix-1, iy  )) + 
          -b1 * make_float3(tex2D(texSRC, ix-1, iy+1)) +
          +b1 * make_float3(tex2D(texSRC, ix+1, iy-1)) +
          +b0 * make_float3(tex2D(texSRC, ix+1, iy  )) + 
          +b1 * make_float3(tex2D(texSRC, ix+1, iy+1)));

    float3 g2 = 0.5f * (
          -b1 * make_float3(tex2D(texSRC, ix-1, iy-1)) + 
          -b0 * make_float3(tex2D(texSRC, ix,   iy-1)) + 
          -b1 * make_float3(tex2D(texSRC, ix+1, iy-1)) +
          +b1 * make_float3(tex2D(texSRC, ix-1, iy+1)) +
          +b0 * make_float3(tex2D(texSRC, ix,   iy+1)) + 
          +b1 * make_float3(tex2D(texSRC, ix+1, iy+1)));
    
    dst(ix, iy) = make_float4( dot(g1, g1), 
                               dot(g1, g2),
                               dot(g2, g2), 1);
}


gpu_image<float4> gpu_cef_scharr( const gpu_image<float4>& src ) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC, src);
    cef_scharr<<<src.blocks(), src.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void cef_jacobi_step( gpu_plm2<float4> dst ) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) return;

    float4 st = tex2D(texST, ix, iy);
    if (st.w < 1) {
        st = make_float4((
                make_float3(tex2D(texST, ix + 1, iy    )) +
                make_float3(tex2D(texST, ix - 1, iy    )) + 
                make_float3(tex2D(texST, ix,     iy + 1)) +
                make_float3(tex2D(texST, ix,     iy - 1))) / 4,
                0);
    }
    dst(ix, iy) = st;
}                          


gpu_image<float4> gpu_cef_jacobi_step( const gpu_image<float4>& st ) {
    gpu_image<float4> dst(st.size());
    bind(&texST, st);
    cef_jacobi_step<<<dst.blocks(), dst.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void cef_restrict( const gpu_plm2<float4> st, 
                              gpu_plm2<float4> dst) 
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) return;

    float4 sum = make_float4(0);
    float4 tmp;
    tmp = st(2*ix,     2*iy    ); if (tmp.w > 0) { sum += tmp; }
    tmp = st(2*ix + 1, 2*iy    ); if (tmp.w > 0) { sum += tmp; }
    tmp = st(2*ix,     2*iy + 1); if (tmp.w > 0) { sum += tmp; }
    tmp = st(2*ix + 1, 2*iy + 1); if (tmp.w > 0) { sum += tmp; }
    if (sum.w > 0) sum /= sum.w;
    dst(ix, iy) = sum;
}


gpu_image<float4> gpu_cef_restrict( const gpu_image<float4>& st ) {
    gpu_image<float4> dst((st.w()+1)/2, (st.h()+1)/2);
    cef_restrict<<<dst.blocks(), dst.threads()>>>(st, dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void cef_interpolate( const gpu_plm2<float4> st_fine, 
                                 gpu_plm2<float4> dst) 
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) return;

    float4 st = st_fine(ix, iy);
    if (st.w < 1) {
        st = make_float4(make_float3(
                tex2D(texST, 0.5f * (ix + 0.5f), 
                             0.5f * (iy + 0.5f) )), 0);
    }
    dst(ix, iy) = st;
}


gpu_image<float4> gpu_cef_interpolate( const gpu_image<float4>& st_fine, const gpu_image<float4>& st_coarse ) {
    gpu_image<float4> dst(st_fine.size());
    bind(&texST, st_coarse);
    texST.filterMode = cudaFilterModeLinear;
    cef_interpolate<<<dst.blocks(), dst.threads()>>>(st_fine, dst);
    texST.filterMode = cudaFilterModePoint;
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_cef_relax( const gpu_image<float4>& st, 
                                 int jacobi_steps ) 
{
    if ((st.w() <= 2) || (st.h() <= 2)) return st;
    gpu_image<float4> tmp;
    tmp = gpu_cef_restrict(st);
    tmp = gpu_cef_relax(tmp, jacobi_steps);
    tmp = gpu_cef_interpolate(st, tmp);
    for (int k = 0; k < jacobi_steps; ++k) {
        tmp = gpu_cef_jacobi_step(tmp);
    }
    return tmp;
}


__global__ void cef_merge( const gpu_plm2<float4> st_cur, const gpu_plm2<float4> st_prev, 
                           float threshold, gpu_plm2<float4> dst ) 
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float4 st = st_cur(ix, iy);
    float mag = st_lambda1(st);
    if (mag < threshold) {
        if (st_prev.ptr) {
            st = st_prev(ix, iy);
        } else {
            st.w = 0;
        }
    } else {
        st.w = 1;
    }
    dst(ix, iy) = st;
}


gpu_image<float4> gpu_cef_merge( const gpu_image<float4>& st_cur, const gpu_image<float4>& st_prev, 
                                 float threshold, int jacobi_steps ) 
{
    gpu_image<float4> dst(st_cur.size());
    cef_merge<<<st_cur.blocks(), st_cur.threads()>>>(st_cur, st_prev, threshold, dst);
    GPU_CHECK_ERROR();
    if (!st_prev.is_valid()) {
        dst = gpu_cef_relax(dst, jacobi_steps);
    }
    return dst;
}


gpu_image<float4> gpu_cef_st( const gpu_image<float4>& src, const gpu_image<float4>& st_prev, 
                              float sigma_d, float tau_r, int jacobi_steps )
{
    gpu_image<float4> st = gpu_cef_scharr(src);
    st = gpu_cef_merge(st, st_prev, tau_r, jacobi_steps);
    st = gpu_gauss_filter_xy(st, sigma_d);
    return st;
}



__global__ void cef_flog( const gpu_plm2<float4> st, float sigma, 
                          gpu_plm2<float> dst ) 
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) return;

    float2 n = st_major_ev(st(ix, iy));
    float2 nabs = fabs(n);
    float ds = 1.0f / ((nabs.x > nabs.y)? nabs.x : nabs.y);
    float2 uv = make_float2(ix + 0.5f, iy + 0.5f);

    float halfWidth = 5 * sigma;
    float sigma2 = sigma * sigma;
    float twoSigma2 = 2 * sigma2;

    float sum = -sigma2 * tex2D(texL, ix + 0.5f, iy + 0.5f);
    for( float d = ds; d <= halfWidth; d += ds ) {
        float k = (d*d - sigma2) * __expf( -d*d / twoSigma2 ); 
        float2 o = d*n;
        float c = tex2D(texL, uv.x - o.x, uv.y - o.y) + 
                  tex2D(texL, uv.x + o.x, uv.y + o.y);
        sum += k * c;
    }

    sum = sum / (sqrtf(2*CUDART_PI_F) * sigma2 * sigma);
    dst(ix, iy) = sum;
}


gpu_image<float> gpu_cef_flog( const gpu_image<float>& L, const gpu_image<float4>& st, float sigma) {
    gpu_image<float> dst(L.size());
    bind(&texL, L);
    texL.filterMode = cudaFilterModeLinear;
    cef_flog<<<dst.blocks(), dst.threads()>>>( st, sigma, dst );
    texL.filterMode = cudaFilterModePoint;
    GPU_CHECK_ERROR();
    return dst;
}


enum minmax_t { MIN_FLT, MAX_FLT };

struct minmax_impl_t {
    float2 uv_;    
    float2 p_;
    float v_;

    __device__ minmax_impl_t(float2 uv) {
        uv_ = p_= uv;
        v_ = tex2D(texL, uv.x, uv.y);
    }

    template <minmax_t T> 
    __device__ void add(float2 p) {
        float L = tex2D(texL, p.x, p.y);
        if ((T == MAX_FLT) && (L > v_)) { p_ = p; v_ = L; }
        if ((T == MIN_FLT) && (L < v_)) { p_ = p; v_ = L; }
    }

    template <minmax_t T> 
    __device__  void run( float2 n, float radius ) {
        float ds;
        float2 dp;
        
        float2 nabs = fabs(n);
        if (nabs.x > nabs.y) {
            ds = 1.0f / nabs.x;
            dp = make_float2(0, 0.5f - 1e-3);
        } else {
            ds = 1.0f / nabs.y;
            dp = make_float2(0.5f - 1e-3, 0);
        }

        for( float d = ds; d <= radius; d += ds ) {
            float2 o = d*n;
            add<T>(uv_ + o + dp); add<T>(uv_ + o - dp);
            add<T>(uv_ - o + dp); add<T>(uv_ - o - dp);
        }
    }
};


__global__ void cef_shock( const gpu_plm2<float4> st, 
                           const gpu_plm2<float> sign, 
                           float radius, float tau,
                           gpu_plm2<float4> dst ) 
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) return;

    minmax_impl_t mm(make_float2(ix + 0.5f, iy + 0.5f));
    float2 n = st_major_ev(st(ix, iy));
    float s = sign(ix, iy);
    if (s < -tau) 
        mm.run<MAX_FLT>(n, radius);
    else if (s > tau) {
        mm.run<MIN_FLT>(n, radius);
    }
    dst(ix, iy) = tex2D(texSRC, mm.p_.x, mm.p_.y);
}


gpu_image<float4> gpu_cef_shock( const gpu_image<float>& L, const gpu_image<float4>& st, 
                                 const gpu_image<float>& sign, const gpu_image<float4>& src,
                                 float radius, float tau )
{
    gpu_image<float4> dst(src.size());
    bind(&texL, L);
    bind(&texSRC, src);
    cef_shock<<<dst.blocks(), dst.threads()>>>( st, sign, radius, tau, dst );
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_ivacef( const gpu_image<float4>& src, int N, float sigma_d, float tau_r, 
                              float sigma_t, float sigma_i, float sigma_g, 
                              float radius, float tau_s, float sigma_a )
{
    gpu_image<float4> img = src;
    gpu_image<float4> st;

    for (int k = 0; k < N; ++k) {
        st = gpu_cef_st(img, st, sigma_d, tau_r, 1);
        img = gpu_stgauss3_filter(img, st, sigma_t, true, true, 2, 1, true);

        st = gpu_cef_st(img, st, sigma_d, tau_r, 1);
        gpu_image<float> L = gpu_rgb2gray(img);
        L =  gpu_gauss_filter_xy(L, sigma_i);
        gpu_image<float> sign = gpu_cef_flog(L, st, sigma_g);
        img = gpu_cef_shock(L, st, sign, src, radius, tau_s);
    }

    img = gpu_stgauss3_filter(img, st, sigma_a, true, true, 2, 1, false);
    return img;
}
