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
#define __STDC_CONSTANT_MACROS
#include "libav_decoder.h"
#include <vector>
#include <algorithm>
#include <cassert>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
#include <libswscale/swscale.h>
}


struct libav_decoder::impl {
    bool done;
    AVFormatContext *formatCtx;
    int streamId;
    AVStream *stream;
    AVCodecContext *codecCtx;
    AVCodec *codec;
    AVFrame *frame0;
    AVFrame *frame1;
    int width;
    int height;
    uint8_t *buffer;
    SwsContext *swsCtx;
    std::vector<int64_t> pts;
    int64_t currentTime;
    int currentFrame;

    impl() {
        done = false;
        formatCtx = 0;
        stream = 0;
        streamId = -1;
        codecCtx = 0;
        codec = 0;
        frame0 = frame1 = 0;
        width = height = 0;
        buffer = 0;
        swsCtx = 0;
        currentTime = AV_NOPTS_VALUE;
        currentFrame = -1;
    }


    ~impl() {
        done = false;
        width = height = 0;

        if (swsCtx) {
            sws_freeContext(swsCtx);
            swsCtx = 0;
        }
        av_freep(&buffer);
        av_freep(&frame0);
        av_freep(&frame1);
        codec = 0;
        if (codecCtx) {
            avcodec_close(codecCtx);
            codecCtx = 0;
        }
        stream = 0;
        if (formatCtx) {
            av_close_input_file(formatCtx);
            formatCtx = 0;
        }
    }
};


libav_decoder* libav_decoder::open(const char* path) {
    static bool s_init = true;
    if (s_init) {
        s_init = false;
        av_register_all();
    }

    impl *m = new impl;
    for (;;) {
        int err;

        #if LIBAVFORMAT_VERSION_MAJOR >= 54
            err = avformat_open_input(&m->formatCtx, path, NULL, NULL);
        #else
            err = av_open_input_file(&m->formatCtx, path, NULL, 0, NULL);
        #endif
        if (err != 0) break;

        #if LIBAVFORMAT_VERSION_MAJOR >= 54
            err = avformat_find_stream_info(m->formatCtx, NULL);
        #else
            err = av_find_stream_info(m->formatCtx);
        #endif
        if (err < 0) break;

        m->streamId = -1;
        for (unsigned i = 0; i < m->formatCtx->nb_streams; ++i) {
            if (m->formatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
                m->streamId = i;
                break;
            }
        }
        if (m->streamId == -1) break;

        m->stream = m->formatCtx->streams[m->streamId];
        m->codecCtx = m->stream->codec;
        m->width = m->codecCtx->width;
        m->height = m->codecCtx->height;
        m->codec = avcodec_find_decoder(m->codecCtx->codec_id);
        if (!m->codec) break;

        #if LIBAVCODEC_VERSION_MAJOR >= 54
            err = avcodec_open2(m->codecCtx, m->codec, NULL);
        #else
            err = avcodec_open(m->codecCtx, m->codec);
        #endif
        if (err != 0) break;

        m->frame0 = avcodec_alloc_frame();
        m->frame1 = avcodec_alloc_frame();
        if (!m->frame0 || !m->frame1) break;

        int frameBytes = avpicture_get_size(PIX_FMT_RGB32, m->width, m->height);
        m->buffer = (uint8_t*)av_malloc(frameBytes * sizeof(uint8_t));
        avpicture_fill((AVPicture*)m->frame1, m->buffer, PIX_FMT_RGB32, m->width, m->height);

        m->swsCtx = sws_getContext( m->width, m->height, m->codecCtx->pix_fmt,
                                    m->width, m->height, PIX_FMT_RGB32,
                                    SWS_POINT, NULL, NULL, NULL );

        AVPacket packet;
        int finished = 0;
        while (av_read_frame(m->formatCtx, &packet) >= 0) {
            if (packet.stream_index == m->streamId) {
                if (!finished) avcodec_decode_video2(m->codecCtx, m->frame0, &finished, &packet);
                if (finished)m->pts.push_back(packet.dts);
            }
            av_free_packet(&packet);
        }
        std::sort(m->pts.begin(), m->pts.end());
        if (m->pts.size() < 1) break;

        if (m->pts.size() == 1) {
            int h = sws_scale(m->swsCtx, m->frame0->data, m->frame0->linesize, 0,
                              m->codecCtx->height, m->frame1->data, m->frame1->linesize);
            m->currentTime = 0;
            m->currentFrame = 0;
            assert(h == m->height);
        } else {
            err = av_seek_frame(m->formatCtx, m->streamId, m->stream->start_time, AVSEEK_FLAG_BACKWARD);
            if (err < 0) break;
            avcodec_flush_buffers(m->codecCtx);
        }

        return new libav_decoder(m);
    }
    delete m;
    return 0;
}


libav_decoder::libav_decoder(impl *p) : m(p) {}


libav_decoder::~libav_decoder() {
    delete m;
    m = 0;
}


bool libav_decoder::next() {
    if (m->done) return false;
    if (m->pts.size() == 1) {
        m->done = true;
        return true;
    }

    int finished = 0;
    AVPacket packet;
    for (;;) {
        if (av_read_frame(m->formatCtx, &packet) >= 0) {
            if (packet.stream_index == m->streamId) {
                avcodec_decode_video2(m->codecCtx, m->frame0, &finished, &packet);
                m->currentTime = packet.dts;
                m->currentFrame = this->frame_from_time(m->currentTime);
                av_free_packet(&packet);
                if (finished) break;
            } else {
                av_free_packet(&packet);
            }
        } else {
            m->done = true;
            break;
        }
    }

    if (finished) {
        int h = sws_scale(m->swsCtx, m->frame0->data, m->frame0->linesize, 0,
                          m->codecCtx->height, m->frame1->data, m->frame1->linesize);
        assert(h == m->height);
    }

    return finished != 0;
}


void libav_decoder::rewind() {
    this->set_frame(0);
}


void libav_decoder::set_time( int64_t ts, bool exact ) {
    int err = av_seek_frame(m->formatCtx, m->streamId, ts, AVSEEK_FLAG_BACKWARD);
    assert(err >= 0);
    avcodec_flush_buffers(m->codecCtx);
    m->done = false;
    m->currentTime = AV_NOPTS_VALUE;
    m->currentFrame = -1;
    if (exact) {
        do {
            next();
        } while (this->current_time() < ts);
    }
}


void libav_decoder::set_frame( int frame, bool exact ) {
    if (this->current_frame() != frame - 1) {
        this->set_time(time_from_frame(frame), false);
    }
    if (exact) {
        do {
            next();
        } while (this->current_frame() < frame);
    }
}


bool libav_decoder::at_end() const {
    return m->done;
}


int libav_decoder::width() const {
    return m->width;
}


int libav_decoder::height() const {
    return m->height;
}


libav_decoder::int64_t libav_decoder::duration() const {
    return m->stream->duration;
}


libav_decoder::int64_t libav_decoder::start_time() const {
    return m->pts[0];
}


libav_decoder::int64_t libav_decoder::end_time() const {
    return m->pts[0] + m->stream->duration;
}


int libav_decoder::frame_count() const {
    return (int)m->pts.size();
}


libav_decoder::rational libav_decoder::frame_rate() const {
    return rational(m->stream->r_frame_rate.num, m->stream->r_frame_rate.den);
}


libav_decoder::rational libav_decoder::time_base() const {
    return rational(m->stream->time_base.num, m->stream->time_base.den);
}


libav_decoder::uint8_t* libav_decoder::buffer() const {
    return m->buffer;
}


libav_decoder::int64_t libav_decoder::current_time() const {
    return m->currentTime;
}


int libav_decoder::current_frame() const {
    return m->currentFrame;
}


double libav_decoder::fps() const {
    double delta = (double)m->stream->duration / m->pts.size();
    return (double)m->stream->time_base.den / (double)m->stream->time_base.num / delta;
}


int libav_decoder::frame_from_time(int64_t ts) const {
    if (m->currentTime == AV_NOPTS_VALUE)
        return -1;
    std::vector<int64_t>::iterator i = std::lower_bound(m->pts.begin(), m->pts.end(), ts);
    if (i == m->pts.end()) return (int)m->pts.size() - 1;
    return i - m->pts.begin();
}


libav_decoder::int64_t libav_decoder::time_from_frame(int frame) const {
    frame = std::min<int>( std::max<int>(0, frame), (int)m->pts.size() - 1 );
    return m->pts[frame];
}


libav_decoder::int64_t libav_decoder::ticks() const {
    return av_gettime();
}


libav_decoder::int64_t libav_decoder::ticks_from_time(int64_t x) const {
    AVRational Q = {1, AV_TIME_BASE};
    return av_rescale_q(x, m->stream->time_base, Q);
}


libav_decoder::int64_t libav_decoder::time_from_ticks(int64_t t) const {
    AVRational Q = {1, AV_TIME_BASE};
    return av_rescale_q(t, Q, m->stream->time_base);
}
