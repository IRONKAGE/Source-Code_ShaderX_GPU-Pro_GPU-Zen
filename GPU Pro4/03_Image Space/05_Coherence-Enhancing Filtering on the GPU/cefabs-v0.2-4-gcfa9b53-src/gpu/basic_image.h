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

#include "gpu_math.h"
#include "gpu_cache.h"
#include <cassert>

enum pixel_type_t {
    PIXEL_TYPE_UCHAR  = 0x11,
    PIXEL_TYPE_UCHAR2 = 0x12,
    PIXEL_TYPE_UCHAR4 = 0x14, 
    PIXEL_TYPE_FLOAT  = 0x21,
    PIXEL_TYPE_FLOAT2 = 0x22,
    PIXEL_TYPE_FLOAT4 = 0x24
};

template <typename T> unsigned pixel_type_id();
template <> inline unsigned pixel_type_id<unsigned char>() { return PIXEL_TYPE_UCHAR; }
template <> inline unsigned pixel_type_id<uchar2>() { return PIXEL_TYPE_UCHAR2; }
template <> inline unsigned pixel_type_id<uchar4>() { return PIXEL_TYPE_UCHAR4; }
template <> inline unsigned pixel_type_id<float>() { return PIXEL_TYPE_FLOAT; }
template <> inline unsigned pixel_type_id<float2>() { return PIXEL_TYPE_FLOAT2; }
template <> inline unsigned pixel_type_id<float4>() { return PIXEL_TYPE_FLOAT4; }


template <typename A> class basic_image_data {
public:
    basic_image_data( unsigned type_id, unsigned type_size, void *ptr, unsigned pitch, 
                      unsigned w, unsigned h, A allocator=A() )
        : m_nrefs(1), 
          m_type_id(type_id), 
          m_type_size(type_size), 
          m_ptr(ptr), 
          m_pitch(pitch), 
          m_w(w), 
          m_h(h), 
          m_shared(ptr != 0),
          m_allocator(allocator)
    {
        if (!m_shared) {
            m_allocator.alloc(&m_ptr, &m_pitch, m_type_size * m_w, m_h);
        }
    }

    void add_ref() {
        ++m_nrefs;
    }

    void release() {
        --m_nrefs;
        if (m_nrefs == 0) delete this;
    }
    
    unsigned type_id() const { return m_type_id; }
    unsigned type_size() const { return m_type_size; }
    void* ptr() const { return m_ptr; }
    unsigned pitch() const { return m_pitch; }
    unsigned w() const { return m_w; }
    unsigned h() const { return m_h; }

protected:
    unsigned m_nrefs;
    unsigned m_type_id;
    unsigned m_type_size;
    void *m_ptr;
    unsigned m_pitch;
    unsigned m_w;
    unsigned m_h;
    bool m_shared;
    A m_allocator;

    ~basic_image_data() {
        if (!m_shared) {
            m_allocator.free(m_ptr, m_pitch, m_type_size * m_w, m_h);
        }
        m_ptr = 0;
        m_pitch = 0;
    }

private:
    basic_image_data(const basic_image_data&);
    const basic_image_data& operator=(const basic_image_data&);
};


template <typename T, typename A> class basic_image {
public:
    typedef basic_image_data<A> data_type;

    basic_image() : m(0) 
    { }

    basic_image(unsigned w, unsigned h, A allocator=A()) {
        m = new data_type(pixel_type_id<T>(), sizeof(T), 0, 0, w, h, allocator);
    } 

    basic_image(uint2 size, A allocator=A()) {
        m = new data_type(pixel_type_id<T>(), sizeof(T), 0, 0, size.x, size.y, allocator);
    } 

    basic_image(data_type *data) {
        m = 0;
        if (data && (data->type_id() == pixel_type_id<T>())) {
            data->add_ref();
            m = data;
        }
    } 

    basic_image(const basic_image& img) {
        if (img.m) img.m->add_ref();
        m = img.m;
    }

    ~basic_image() {
        if (m) m->release();
        m = 0;
    }

    const basic_image& operator=(const basic_image& img) {
        if (img.m) img.m->add_ref();
        if (m) m->release();
        m = img.m;
        return *this;
    }

    void swap(basic_image& img) {
        data_type *tmp = m;
        m = img.m;
        img.m = tmp;
    }

    data_type* data() const { return m; }
    bool is_valid() const { return m && m->w() && m->h(); }

    const T* ptr() const { return (T*)m->ptr(); }
    T* ptr() { return (T*)m->ptr(); }
    const char* ptr8u() const { return (char*)m->ptr(); }
    char* ptr8u() { return (char*)m->ptr(); }
    unsigned pitch() const { return m->pitch(); }
    unsigned w() const { return m->w(); }
    unsigned h() const { return m->h(); }
    uint2 size() const { return make_uint2(m->w(), m->h()); }

private:
    data_type *m;
};
