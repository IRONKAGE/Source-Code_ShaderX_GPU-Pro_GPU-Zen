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
#include "imageutil.h"
#include "gpu_convert.h"


cpu_image<uchar4> cpu_image_from_qimage(const QImage& image) {
    if (!image.isNull()) {
        QImage qi = image;
        if (qi.format() != QImage::Format_RGB32) {
            qi = qi.convertToFormat(QImage::Format_RGB32);
        }
        return cpu_image<uchar4>((const uchar4*)qi.bits(), qi.bytesPerLine(), qi.width(), qi.height());
    }
    return cpu_image<uchar4>();
}


QImage cpu_data_to_qimage(const cpu_image_data *data) {
    if (!data) return QImage();

    if (data->type_id() == pixel_type_id<float>()) {
        QImage img(data->w(), data->h(), QImage::Format_Indexed8);
        for (unsigned i = 0; i < 256; ++i) img.setColor(i, 0xff000000 | (i << 16) | (i << 8) | i);
        const float *p = (const float*)data->ptr(); 
        for (unsigned j = 0; j < data->h(); ++j) {
            unsigned char* q = img.scanLine(j);
            for (unsigned i = 0; i < data->w(); ++i) {
                *q++ = (unsigned char)(255.0f * clamp(*p++, 0.0f, 1.0f));
            }
        }
        return img;
    } 

    if (data->type_id() == pixel_type_id<unsigned char>()) {
        QImage img(data->w(), data->h(), QImage::Format_Indexed8);
        for (unsigned i = 0; i < 256; ++i) img.setColor(i, 0xff000000 | (i << 16) | (i << 8) | i);
        const unsigned char *p = (const unsigned char*)data->ptr(); 
        for (unsigned j = 0; j < data->h(); ++j) {
            unsigned char* q = img.scanLine(j);
            for (unsigned i = 0; i < data->w(); ++i) {
                *q++ = *p++;
            }
        }
        return img;
    } 
    
    if (data->type_id() == pixel_type_id<float4>()) {
        QImage img(data->w(), data->h(), QImage::Format_RGB32);
        const unsigned N = data->w() * data->h();
        const float4 *p = (const float4*)data->ptr(); 
        uchar4* q = (uchar4*)img.bits();
        for (unsigned i = 0; i < N; ++i) {
            q->x = (unsigned char)(255.0f * clamp(p->x, 0.0f, 1.0f));
            q->y = (unsigned char)(255.0f * clamp(p->y, 0.0f, 1.0f));
            q->z = (unsigned char)(255.0f * clamp(p->z, 0.0f, 1.0f));
            q->w = 255;
            p++;
            q++;
        }
        return img;
    }
    
    if (data->type_id() == pixel_type_id<uchar4>()) {
        QImage img(data->w(), data->h(), QImage::Format_RGB32);
        const unsigned N = data->w() * data->h();
        const uchar4 *p = (const uchar4*)data->ptr(); 
        uchar4* q = (uchar4*)img.bits();
        for (unsigned i = 0; i < N; ++i) {
            q->x = p->x;
            q->y = p->y;
            q->z = p->z;
            q->w = 255;
            p++;
            q++;
        }
        return img;
    }

    return QImage();
}


template <> gpu_image<uchar4> gpu_image_from_qimage(const QImage& image) {
    if (!image.isNull()) {
        return gpu_image<uchar4>((uchar4*)image.bits(), image.bytesPerLine(), image.width(), image.height());
    }
    return gpu_image<uchar4>();
}


template <> gpu_image<float4> gpu_image_from_qimage(const QImage& image) {
    if (!image.isNull()) {
        return gpu_8u_to_32f(gpu_image_from_qimage<uchar4>(image));
    }
    return gpu_image<float4>();
}


QImage gpu_image_to_qimage(const gpu_image<uchar>& image) {
    QImage dst(image.w(), image.h(), QImage::Format_Indexed8);
    dst.setColorCount(256);
    for (int i = 0; i < 256; ++i) dst.setColor(i, qRgb(i,i,i));
    copy(dst.bits(), dst.bytesPerLine(), &image);
    return dst;
}


QImage gpu_image_to_qimage(const gpu_image<uchar4>& image) {
    QImage dst(image.w(), image.h(), QImage::Format_RGB32);
    copy(dst.bits(), dst.bytesPerLine(), &image);
    return dst;
}


QImage gpu_image_to_qimage(const gpu_image<float>& image) {
    return gpu_image_to_qimage(gpu_32f_to_8u(image));
}

QImage gpu_image_to_qimage(const gpu_image<float4>& image) {
    return gpu_image_to_qimage(gpu_32f_to_8u(image));
}
