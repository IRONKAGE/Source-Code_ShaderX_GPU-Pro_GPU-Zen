//
// Copyright (c) 2011 by Jan Eric Kyprianidis <www.kyprianidis.com>
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

class libav_encoder {
public:
    typedef unsigned char uint8_t;
    typedef long long int64_t;
    
    static libav_encoder* create( const char *path, unsigned width, unsigned height,
                                  const std::pair<int,int>& frame_rate, 
                                  int bit_rate = 8000000 );
    ~libav_encoder();
    
    bool    append_frame  ( const uint8_t *buffer );
    void    finish        ();

private:
    struct impl;
    impl *m;
    libav_encoder(impl*);
};
