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
#include "libav_encoder.h"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
#include <libswscale/swscale.h>
}


static bool alloc_frame(AVFrame **frame, uint8_t **buf, enum PixelFormat pix_fmt, int w, int h) {
    *buf = 0;
    *frame = 0;

    AVFrame *f  = avcodec_alloc_frame();
    if (!f)
        return false;

    int size = avpicture_get_size(pix_fmt, w, h);
    uint8_t *b = (uint8_t*)av_malloc(size);
    if (!b) {
        av_free(f);
        return false;
    }
    avpicture_fill((AVPicture*)f, b, pix_fmt, w, h);

    *frame = f;
    *buf = b;
    return true;
}


struct libav_encoder::impl {
    AVOutputFormat *format;
    AVFormatContext *formatCtx;
    AVStream *stream;
    AVCodecContext *codecCtx;
    AVFrame *frame0;
    uint8_t *buffer0;
    AVFrame *frame1;
    uint8_t *buffer1;
    uint8_t *output_buffer;
    int output_size;
    SwsContext *swsCtx;

    impl() {
        format = 0;
        formatCtx = 0;
        stream = 0;
        codecCtx = 0;
        frame0 = 0;
        buffer0 = 0;
        frame1 = 0;
        buffer1 = 0;
        output_buffer = 0;
        output_size = 0;
        swsCtx = 0;
    }

    ~impl() {
        if (swsCtx)
            sws_freeContext(swsCtx);

        if (codecCtx)
            avcodec_close(codecCtx);

        if (buffer0)
            av_free(buffer0);
        if (frame0)
            av_free(frame0);
        if (buffer1)
            av_free(buffer1);
        if (frame1)
            av_free(frame1);
        if (output_buffer)
            av_free(output_buffer);

        if (formatCtx) {
            for(unsigned i = 0; i < formatCtx->nb_streams; i++) {
                av_freep(&formatCtx->streams[i]->codec);
                av_freep(&formatCtx->streams[i]);
            }

            if (formatCtx->pb) {
                #if LIBAVFORMAT_VERSION_MAJOR >= 54
                    avio_close(formatCtx->pb);
                #else
                    url_fclose(formatCtx->pb);
                #endif
            }

            av_free(formatCtx);
        }
    }
};


libav_encoder* libav_encoder::create( const char *path, unsigned width, unsigned height,
                                      const std::pair<int, int>& fps, int bit_rate )
{
    impl *m = new impl;
    for (;;) {
        m->format = av_guess_format(NULL, path, NULL);
        if (!m->format)
            break;

        if ((m->format->flags & AVFMT_NOFILE) || (m->format->video_codec == CODEC_ID_NONE))
            break;

        m->formatCtx = avformat_alloc_context();
        if (!m->formatCtx)
            break;
        m->formatCtx->oformat = m->format;

        {
            #if LIBAVFORMAT_VERSION_MAJOR >= 54
                m->stream = avformat_new_stream(m->formatCtx, 0);
            #else
                m->stream = av_new_stream(m->formatCtx, 0);
            #endif
            if (!m->stream) {
                fprintf(stderr, "Could not alloc stream\n");
                break;
            }
            m->codecCtx = m->stream->codec;

            m->codecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
            if (strcmp(m->format->name, "avi") == 0) {
                m->codecCtx->codec_id = CODEC_ID_RAWVIDEO;
            } else {
                m->codecCtx->codec_id = m->format->video_codec;
            }

            AVCodec *codec = avcodec_find_encoder(m->codecCtx->codec_id);
            if (!codec)
                break;

            if (strcmp(m->format->name, "avi") == 0) {
                m->codecCtx->pix_fmt = PIX_FMT_BGR24;
            } else {
                if (codec->pix_fmts)
                    m->codecCtx->pix_fmt = codec->pix_fmts[0];
                else
                    m->codecCtx->pix_fmt = PIX_FMT_YUV420P;
            }

            m->codecCtx->bit_rate = 8000000;
            m->codecCtx->width = width;   /* resolution must be a multiple of two */
            m->codecCtx->height = height;
            m->codecCtx->time_base.num = fps.second;
            m->codecCtx->time_base.den = fps.first;
            m->codecCtx->gop_size = 12; /* emit one intra frame every twelve frames at most */

            if(m->formatCtx->oformat->flags & AVFMT_GLOBALHEADER)
                m->codecCtx->flags |= CODEC_FLAG_GLOBAL_HEADER;

                #if LIBAVFORMAT_VERSION_MAJOR >= 54
                    if (avcodec_open2(m->codecCtx, codec, NULL) < 0) break;
                #else
                    if (avcodec_open(m->codecCtx, codec) < 0) break;
                #endif

            if (!(m->formatCtx->oformat->flags & AVFMT_RAWPICTURE)) {
                m->output_size = m->codecCtx->width * m->codecCtx->height * 4;
                m->output_buffer = (uint8_t*)av_malloc(m->output_size);
            }

            if (!alloc_frame(&m->frame1, &m->buffer1, m->codecCtx->pix_fmt, m->codecCtx->width, m->codecCtx->height))
                break;
            if (!alloc_frame(&m->frame0, &m->buffer0, PIX_FMT_RGB32, m->codecCtx->width, m->codecCtx->height))
                break;
        }

        #if LIBAVFORMAT_VERSION_MAJOR >= 54
            if (avio_open(&m->formatCtx->pb, path, AVIO_FLAG_WRITE) < 0) break;
            avformat_write_header(m->formatCtx, NULL);
        #else
            if (url_fopen(&m->formatCtx->pb, path, URL_WRONLY) < 0) break;
            av_write_header(m->formatCtx);
        #endif

        m->swsCtx = sws_getContext( m->codecCtx->width, m->codecCtx->height, PIX_FMT_RGB32,
                                    m->codecCtx->width, m->codecCtx->height, m->codecCtx->pix_fmt,
                                    SWS_POINT, NULL, NULL, NULL );
        if (!m->swsCtx)
            break;

        return new libav_encoder(m);
    }
    delete m;
    return 0;
}


libav_encoder::libav_encoder(impl *p) : m(p) {}


libav_encoder::~libav_encoder() {
}


bool libav_encoder::append_frame( const uint8_t *buffer ) {
    int out_size, ret;
    AVCodecContext *c;

    c = m->stream->codec;
    memcpy(m->buffer0, buffer, 4 * m->codecCtx->width * m->codecCtx->height);
    sws_scale(m->swsCtx, m->frame0->data, m->frame0->linesize,
              0, m->codecCtx->height, m->frame1->data, m->frame1->linesize);

    if (m->formatCtx->oformat->flags & AVFMT_RAWPICTURE) {
        AVPacket pkt;
        av_init_packet(&pkt);
        pkt.flags |= AV_PKT_FLAG_KEY;
        pkt.stream_index = m->stream->index;
        pkt.data= (uint8_t*)m->frame1;
        pkt.size= sizeof(AVPicture);

        ret = av_interleaved_write_frame(m->formatCtx, &pkt);
    } else {
        out_size = avcodec_encode_video(m->codecCtx, m->output_buffer, m->output_size, m->frame1);
        if (out_size > 0) {
            AVPacket pkt;
            av_init_packet(&pkt);

            if (m->codecCtx->coded_frame->pts != AV_NOPTS_VALUE)
                pkt.pts= av_rescale_q(m->codecCtx->coded_frame->pts, c->time_base, m->stream->time_base);
            if (m->codecCtx->coded_frame->key_frame)
                pkt.flags |= AV_PKT_FLAG_KEY;
            pkt.stream_index= m->stream->index;
            pkt.data = m->output_buffer;
            pkt.size = out_size;

            ret = av_interleaved_write_frame(m->formatCtx, &pkt);
        } else {
            ret = 0;
        }
    }
    return (ret != 0);
}


void libav_encoder::finish() {
    av_write_trailer(m->formatCtx);
    delete m;
    m = 0;
}
