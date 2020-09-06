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
#include <vector>

gpu_image<float4> gpu_stgauss3_filter( const gpu_image<float4>& src, const gpu_image<float4>& st,
                                       float sigma, bool src_linear, bool st_linear,
                                       int order, float step_size, bool adaptive );

std::vector<float3> gpu_stgauss3_path( int ix, int iy, const cpu_image<float4>& st,
                                       float sigma, bool st_linear,
                                       int order, float step_size, bool adaptive );


template <typename ST, typename F>
inline __host__ __device__ void st_integrate_euler( float2 p0, const ST& st, F& f, unsigned w, unsigned h,
                                                    float step_size, bool adaptive )
{
    f(0, p0);
    float2 v0 = st_minor_ev(st(p0.x, p0.y));
    float sign = -1;
    float dr = f.radius() / CUDART_PI_F;
    do {
        float2 v = v0 * sign;
        float2 p = p0;
        float u = 0;

        for (;;) {

            float2 t = st_minor_ev(st(p.x, p.y));
            float vt = dot(v, t);
            if (vt < 0) {
                t = -t;
                vt = -vt;
            }

            v = t;
            p += step_size * t;

            if (adaptive) {
                float Lk = dr * acosf(fminf(vt,1));
                u += fmaxf(step_size, Lk);
            } else {
                u += step_size;
            }

            if ((u >= f.radius()) || (p.x < 0) || (p.x >= w) ||
                (p.y < 0) || (p.y >= h)) break;

            f(copysignf(u, sign), p);
        }

        sign *= -1;
    } while (sign > 0);
}


template <typename ST, typename F> inline __host__ __device__
void st_integrate_rk2( float2 p0, const ST& st, F& f,
                       unsigned w, unsigned h,
                       float step_size, bool adaptive )
{
    f(0, p0);
    float2 v0 = st_minor_ev(st(p0.x, p0.y));
    float sign = -1;
    float dr = (f.radius() /*+ 0.5f * step_size*/) / CUDART_PI_F;
    do {
        float2 v = v0 * sign;
        float2 p = p0;
        float u = 0;

        for (int kk = 0; kk < 100; ++kk) {
            float2 t = st_minor_ev(st(p.x, p.y));
            if (dot(v, t) < 0) t = -t;

            float2 ph = p + 0.5f * step_size * t;
            t = st_minor_ev(st(ph.x, ph.y));
            float vt = dot(v, t);
            if (vt < 0) {
                t = -t;
                vt = -vt;
            }

            v = t;
            p += step_size * t;

            if (adaptive) {
                float Lk = dr * acosf(fminf(vt,1));
                u += fmaxf(step_size, Lk);
            } else {
                u += step_size;
            }

            if ((u >= f.radius()) || (p.x < 0) || (p.x >= w) ||
                (p.y < 0) || (p.y >= h)) break;

            f(copysignf(u, sign), p);
        }

        sign *= -1;
    } while (sign > 0);
}
