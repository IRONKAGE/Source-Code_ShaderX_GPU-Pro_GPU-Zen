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
#pragma once

#include "gpu_image.h"

gpu_image<float> gpu_srgb2linear( const gpu_image<float>& src);
gpu_image<float> gpu_linear2srgb( const gpu_image<float>& src);

gpu_image<float4> gpu_rgb2lab( const gpu_image<float4>& src);
gpu_image<float4> gpu_lab2rgb( const gpu_image<float4>& src);

gpu_image<float4> gpu_gray2rgb( const gpu_image<float>& src, bool saturate=true );
gpu_image<float> gpu_rgb2gray( const gpu_image<float4>& src );

gpu_image<float4> gpu_swap_rgba( const gpu_image<float4>& src );

gpu_image<float4> gpu_colorize_sign( const gpu_image<float>& src, float scale );
