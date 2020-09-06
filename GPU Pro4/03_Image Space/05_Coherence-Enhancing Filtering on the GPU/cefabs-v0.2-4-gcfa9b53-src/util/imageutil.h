//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2012 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
#pragma once

#include "gpu_image.h"

cpu_image<uchar4> cpu_image_from_qimage(const QImage& image);

QImage cpu_data_to_qimage(const cpu_image_data *data);
template <typename T>
inline QImage cpu_image_to_qimage(const cpu_image<T>& image) {
    return cpu_data_to_qimage(image.data());
}

template <typename T> gpu_image<T> gpu_image_from_qimage(const QImage& image);
template <> gpu_image<uchar4> gpu_image_from_qimage(const QImage& image);
template <> gpu_image<float4> gpu_image_from_qimage(const QImage& image);

QImage gpu_image_to_qimage(const gpu_image<uchar>& image);
QImage gpu_image_to_qimage(const gpu_image<uchar4>& image);
QImage gpu_image_to_qimage(const gpu_image<float>& image);
QImage gpu_image_to_qimage(const gpu_image<float4>& image);
