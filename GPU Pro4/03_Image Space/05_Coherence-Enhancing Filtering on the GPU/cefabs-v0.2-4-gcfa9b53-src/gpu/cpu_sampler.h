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

template<typename T> 
struct cpu_sampler {
    const cpu_image<T>& img_;
    cudaTextureFilterMode filter_mode_;

    cpu_sampler(const cpu_image<T>& img, cudaTextureFilterMode filter_mode=cudaFilterModePoint) 
        : img_(img), filter_mode_(filter_mode)
    { }

    unsigned w() const {
        return img_.w(); 
    }
    
    unsigned h() const {
        return img_.w(); 
    }

    T operator()(float x, float y) const { 
        return (filter_mode_ == cudaFilterModePoint)? img_(x, y) : img_.sample_linear(x, y);
    } 
};
