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


inline __host__ __device__
void solve_eig_psd( float E, float F, float G, float& lambda1,
                    float& lambda2, float2& ev )
{
    float B = (E + G) / 2;
    if (B > 0) {
        float D = (E - G) / 2;
        float FF = F*F;
        float R = sqrtf(D*D + FF);
        lambda1 = B + R;
        lambda2 = fmaxf(0, E*G - FF) / lambda1;

        if (R > 0) {
            if (D >= 0) {
                float nx = D + R;
                ev = make_float2(nx, F) * rsqrtf(nx*nx + FF);
            } else {
                float ny = -D + R;
                ev = make_float2(F, ny) * rsqrtf(FF + ny*ny);
            }
        } else {
            ev = make_float2(1, 0);
        }
    } else {
        lambda1 = lambda2 = 0;
        ev = make_float2(1, 0);
    }
}


inline __host__ __device__
float2 solve_eig_psd_ev( float E, float F, float G )
{
    float B = (E + G) / 2;
    if (B > 0) {
        float D = (E - G) / 2;
        float FF = F*F;
        float R = sqrtf(D*D + FF);

        if (R > 0) {
            if (D >= 0) {
                float nx = D + R;
                return make_float2(nx, F) * rsqrtf(nx*nx + FF);
            } else {
                float ny = -D + R;
                return make_float2(F, ny) * rsqrtf(FF + ny*ny);
            }
        }
    }
    return make_float2(1, 0);
}


inline __host__ __device__
float solve_eig_psd_lambda1( float E, float F, float G ) {
    float B = (E + G) / 2;
    if (B > 0) {
        float D = (E - G) / 2;
        float FF = F*F;
        float R = sqrtf(D*D + FF);
        return B + R;
    }
    return 0;
}


inline __host__ __device__ float2 st_major_ev(const float4 g) {
    return solve_eig_psd_ev(g.x, g.y, g.z);
}


inline __host__ __device__ float2 st_minor_ev(const float4 g) {
    float2 ev = solve_eig_psd_ev(g.x, g.y, g.z);
    return make_float2(ev.y, -ev.x);
}


inline __host__ __device__ float st_lambda1(float4 g) {
    return solve_eig_psd_lambda1(g.x, g.y, g.z);
}
