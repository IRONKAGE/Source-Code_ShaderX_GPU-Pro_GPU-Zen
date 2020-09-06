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

#include "gpu_image.h"

gpu_image<float4> gpu_cef_scharr( const gpu_image<float4>& src );

gpu_image<float4> gpu_cef_restrict( const gpu_image<float4>& src );

gpu_image<float4> gpu_cef_interpolate( const gpu_image<float4>& src0, const gpu_image<float4>& src1 );

gpu_image<float4> gpu_cef_relax( const gpu_image<float4>& st, int jacobi_steps );

gpu_image<float4> gpu_cef_merge( const gpu_image<float4>& st_cur, const gpu_image<float4>& 
                                 st_prev, float threshold, int jacobi_steps );

gpu_image<float4> gpu_cef_st( const gpu_image<float4>& src, const gpu_image<float4>& st, 
                              float sigma_d, float tau_r, int jacobi_steps );

gpu_image<float> gpu_cef_flog( const gpu_image<float>& L, const gpu_image<float4>& st, 
                               float sigma );

gpu_image<float4> gpu_cef_shock( const gpu_image<float>& L, const gpu_image<float4>& st, 
                                 const gpu_image<float>& sign, const gpu_image<float4>& src,
                                 float radius, float tau );

gpu_image<float4> gpu_ivacef( const gpu_image<float4>& src, int N=5, 
                              float sigma_d=1, float tau_r=0.002f, float sigma_t=6, 
                              float max_angle=22.5f, float sigma_i=0, float sigma_g=1.5f, 
                              float radius=2, float tau_s=0.005f, float sigma_a=1.5f );

