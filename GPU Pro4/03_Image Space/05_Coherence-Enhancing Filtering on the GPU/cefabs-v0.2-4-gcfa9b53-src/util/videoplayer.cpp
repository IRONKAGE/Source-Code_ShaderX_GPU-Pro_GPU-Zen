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
#include "videoplayer.h"
#ifdef HAVE_LIBAV
#include "libav_decoder.h"
#include "libav_encoder.h"


class VideoPlayer::Thread : public QThread {
public:
    libav_decoder *m_libav;
    QMutex m_decoderMutex;
    QWaitCondition m_wait;
    bool m_stop;
    int m_requestedFrame;
    int m_activeFrame;
    QMutex m_cacheMutex;
    int m_cacheMin;
    int m_cacheMax;
    int m_cacheRadius;
    QContiguousCache<QImage> m_cache;

    Thread(VideoPlayer *player, libav_decoder *libav, int cacheRadius, int cacheMin, int cacheMax ) 
        : QThread(player), m_cache(cacheMax)
    {
        m_cacheRadius = cacheRadius;
        m_cacheMin = cacheMin;
        m_cacheMax = cacheMax;
        m_libav = libav;
        m_stop = false;
        m_requestedFrame = 0;
        m_activeFrame = -1;
    }

    ~Thread() {
        if (isRunning()) {
            m_stop = true;
            m_decoderMutex.lock();
            m_wait.wakeOne();
            m_decoderMutex.unlock();
            wait();
        }
        assert(!isRunning());
        m_cache.clear();
        if (m_libav) {
            delete m_libav;
            m_libav = 0;
        }
    }

    VideoPlayer* parent() const { 
        return (VideoPlayer*)QObject::parent(); 
    }

    virtual void run() {
        while (!m_stop) {
            m_decoderMutex.lock();

            int reqMin = qMax(m_requestedFrame - m_cacheRadius, 0);
            int reqMax = qMin(m_requestedFrame + m_cacheRadius, m_libav->frame_count() - 1);
            int reqMax2 = qMin(m_requestedFrame + m_cacheMin, m_libav->frame_count() - 1);

            bool decodeNext = !m_libav->at_end() && 
                ((m_cache.available() > 0) || !m_cache.containsIndex(reqMax2));

            if (decodeNext || (m_activeFrame < 0)) {
                if (decodeNext) {
                    m_libav->next();
                    QImage img(m_libav->buffer(), m_libav->width(), m_libav->height(), QImage::Format_RGB32);
                    m_cacheMutex.lock();
                    m_cache.insert(m_libav->current_frame(), img.copy(img.rect()));
                    m_cacheMutex.unlock();
                }

                if ((m_activeFrame < 0) && m_cache.containsIndex(reqMin) && m_cache.containsIndex(reqMax)) {
                    m_cacheMutex.lock();
                    m_activeFrame = m_requestedFrame;
                    m_cacheMutex.unlock();
                    QApplication::postEvent(parent(), new QEvent(QEvent::User), Qt::HighEventPriority);
                }
            } else {
                m_wait.wait(&m_decoderMutex);
            }
            m_decoderMutex.unlock();
        }
    }

    void seek(int frame) {
        frame = qBound(0, frame, m_libav->frame_count() - 1);
        if (m_requestedFrame != frame) {
            QMutexLocker lock(&m_decoderMutex);

            m_requestedFrame = frame;
            m_activeFrame = -1;

            int reqMin = qMax(0, m_requestedFrame - m_cacheRadius);
            if (!m_cache.containsIndex(reqMin)) {
                m_cache.clear();
            }

            if (m_cache.isEmpty()) {
                m_libav->set_frame(reqMin, false);
            }

            m_wait.wakeOne();
        }
    }
};
#endif


VideoPlayer::VideoPlayer( QObject *parent, const QString& filename, int cacheRadius, int cacheMin, int cacheMax ) 
    : QObject(parent), m_filename(filename), m_thread(0),
      m_cacheRadius(cacheRadius), m_cacheMin(cacheMin), m_cacheMax(cacheMax)
{
}


VideoPlayer::~VideoPlayer() {
    if (isPlaying()) pause();
}


void VideoPlayer::saveSettings(QSettings& settings) {
    settings.setValue("filename", m_filename); 
}


void VideoPlayer::restoreSettings(QSettings& settings) {
    QString filename = settings.value("filename", m_filename).toString();
    if (QFile::exists(filename)) {
        open(filename);
    } else if (QFile::exists(m_filename)) {
        open(m_filename);
    } 
}


bool VideoPlayer::open(const QString& path) {
    #ifdef HAVE_LIBAV
    if (m_thread) {
        delete m_thread;
        m_thread = 0;
    }
    #endif

    m_currentFrame = -1;
    m_image = QImage();

    QImage tmp(path);
    if (!tmp.isNull()) {
        m_image = tmp.convertToFormat(QImage::Format_RGB32);
        m_currentFrame = 0;
        m_filename = path;
        videoChanged(size());
        videoChanged(frameCount());
        currentFrameChanged(currentFrame());
        currentFrameChanged(image());
        return true;
    }

    #ifdef HAVE_LIBAV
    QByteArray path8 = path.toLocal8Bit();
    libav_decoder *libav = libav_decoder::open(path8.data());
    if (libav) {
        m_thread = new Thread(this, libav, m_cacheRadius, m_cacheMin, m_cacheMax);
        m_filename = path;
    }
    videoChanged(size());
    videoChanged(frameCount());
    if (m_thread) m_thread->start();
    #endif

    return m_thread != 0;
}


void VideoPlayer::open() {
    #ifdef HAVE_LIBAV
    QString filter = "Images and Videos (*.png *.bmp *.jpg *.jpeg *.mov *.mp4 *.m4v *.3gp *.avi; *.wmv);;All files (*.*)";
    #else
    QString filter = "Images (*.png *.bmp);;All files (*.*)";
    #endif
    QString filename = QFileDialog::getOpenFileName(NULL, "Open", m_filename, filter);
    if (!filename.isEmpty()) {
        if (!open(filename)) {
            QMessageBox::critical(NULL, "Error", QString("Loading '%1' failed!").arg(filename));
        }
    }
}


void VideoPlayer::close() {
    #ifdef HAVE_LIBAV
    if (m_thread) {
        delete m_thread;
        m_thread = 0;
    }
    #endif
    m_currentFrame = -1;
    m_image = QImage();

    videoChanged(size());
    videoChanged(frameCount());
    currentFrameChanged(currentFrame());
    currentFrameChanged(image());
}


const QString& VideoPlayer::filename() const {
    return m_filename;
}

    
bool VideoPlayer::isValid() const {
    return (m_thread != 0) || !m_image.isNull();
}


bool VideoPlayer::isBusy() const {
    #ifdef HAVE_LIBAV
    if (m_thread) return m_currentFrame != m_thread->m_requestedFrame;
    #endif
    return false;
}


QSize VideoPlayer::size() const {
    #ifdef HAVE_LIBAV
    if (m_thread) return QSize(m_thread->m_libav->width(), m_thread->m_libav->height());
    #endif
    return m_image.size();
}


int VideoPlayer::width() const {
    #ifdef HAVE_LIBAV
    if (m_thread) return m_thread->m_libav->width();
    #endif
    return m_image.width();
}


int VideoPlayer::height() const {
    #ifdef HAVE_LIBAV
    if (m_thread) return m_thread->m_libav->height();
    #endif
    return m_image.height();
}


int VideoPlayer::frameCount() const {
    #ifdef HAVE_LIBAV
    if (m_thread) return m_thread->m_libav->frame_count();
    #endif
    return m_image.isNull()? 0 : 1;
}


double VideoPlayer::fps() const {
    #ifdef HAVE_LIBAV
    if (m_thread) return m_thread->m_libav->fps();
    #endif
    return 1;
}


QPair<int,int> VideoPlayer::frameRate() const {
    #ifdef HAVE_LIBAV
    if (m_thread) {
        std::pair<int,int> fr = m_thread->m_libav->frame_rate();
        return qMakePair(fr.first, fr.second);
    }
    #endif
    return QPair<int,int>(1,1);
}


QPair<int,int> VideoPlayer::timeBase() const {
    #ifdef HAVE_LIBAV
    if (m_thread) {
        std::pair<int,int> fr = m_thread->m_libav->time_base();
        return qMakePair(fr.first, fr.second);
    }
    #endif
    return QPair<int,int>(1,1);
}


int VideoPlayer::currentFrame() const {
    return m_currentFrame;
}


QImage VideoPlayer::image(int index) const {
    #ifdef HAVE_LIBAV
    if (m_thread && (index != 0)) {
        QMutexLocker lock(&m_thread->m_cacheMutex);
        if (m_currentFrame + index < m_thread->m_cache.firstIndex()) return m_thread->m_cache.first();
        if (m_currentFrame + index > m_thread->m_cache.lastIndex()) return m_thread->m_cache.last();
        return m_thread->m_cache.at(m_currentFrame + index);
    }
    #endif
    return m_image;
}


qint64 VideoPlayer::time(int index) const {
    #ifdef HAVE_LIBAV
    if (m_thread) return m_thread->m_libav->time_from_frame(m_currentFrame + index);
    #endif
    return 0;
}


bool VideoPlayer::isPlaying() const {
    return m_thread && m_timer.isActive();
}


QList<VideoPlayer*> VideoPlayer::slaves() const {
    QList<VideoPlayer*> S;
    QObjectList L = children();
    for (int i = 0; i < L.size(); ++i) {
        VideoPlayer *p = qobject_cast<VideoPlayer*>(L[i]);
        if (p) {
            S.append(p);
        }
    }
    return S;
}


VideoPlayer* VideoPlayer::slave(int index) const {
    QObjectList L = children();
    int n = 0;
    for (int i = 0; i < L.size(); ++i) {
        VideoPlayer *p = qobject_cast<VideoPlayer*>(L[i]);
        if (p) {
            if (n == index) return p;
            ++n;
        }

    }
    return 0;
}


void VideoPlayer::rewind() {
    setCurrentFrame(0);
}


void VideoPlayer::stepForward() {
    #ifdef HAVE_LIBAV
    if (m_thread) setCurrentFrame(m_thread->m_requestedFrame + 1);
    #endif
}


void VideoPlayer::stepBack() {
    #ifdef HAVE_LIBAV
    if (m_thread) setCurrentFrame(m_thread->m_requestedFrame - 1);
    #endif
}


void VideoPlayer::setCurrentFrame(int frame) {
    #ifdef HAVE_LIBAV
    if (m_thread && (m_thread->m_requestedFrame != frame)) {
        m_thread->seek(frame); 
        QObjectList L = children();
        for (int i = 0; i < L.size(); ++i) {
            VideoPlayer *p = qobject_cast<VideoPlayer*>(L[i]);
            if (p) {
                p->setCurrentFrame(frame);
            }
        }
    }
    #endif
}


void VideoPlayer::setPlayback(bool playing) {
    #ifdef HAVE_LIBAV
    if (m_thread && (isPlaying() != playing)) {
        if (playing) {
            libav_decoder *f = m_thread->m_libav;
            m_ticks = f->ticks();
            qint64 t = f->time_from_frame(m_thread->m_requestedFrame);
            m_ticks -= f->ticks_from_time(t);
            m_timer.start(0, this);
            playbackStarted();
            playbackChanged(true);
        } else {
            if (m_timer.isActive()) m_timer.stop();
            playbackPaused();
            playbackChanged(false);
        }
    }
    #endif
}

    
void VideoPlayer::play() {
    setPlayback(true);
}


void VideoPlayer::pause() {
    setPlayback(false);
}


void VideoPlayer::toggle() {
    if (m_thread) {
        if (!isPlaying()) {
            play();
        } else {
            pause();
        }
    }
}


void VideoPlayer::setOutput(const QImage& image) {
    m_output = image;
    outputChanged(m_output);
}


void VideoPlayer::record() {
    #ifdef HAVE_LIBAV
    if (!m_thread || isPlaying()) return;

    QSettings settings;
    QString inputPath = m_filename;
    QString outputPath = settings.value("savename", inputPath).toString();

    QFileInfo fi(inputPath);
	QFileInfo fo(outputPath);
    QString filename = QFileDialog::getSaveFileName(NULL, "Record", fo.dir().filePath(fi.baseName() + "-out"), 
        "MPEG 4 (*.mp4);;Uncompressed AVI (*.avi);;All files (*.*)");
    if (filename.isEmpty())
        return;

    libav_decoder *f = m_thread->m_libav;
    unsigned nframes = f->frame_count();
    libav_encoder *encoder = libav_encoder::create(filename.toStdString().c_str(), f->width(), f->height(), f->frame_rate()); 
    if (!encoder) {
        QMessageBox::critical(NULL, "Error", QString("Creation of %1 failed!").arg(filename));
        return;
    }

    QProgressDialog progress("Recording...", "Abort", 0, nframes-1, NULL);
    progress.setWindowModality(Qt::ApplicationModal);
    progress.show();

    for (unsigned i = 0; i < nframes; ++i) {
        progress.setValue(i);

        setCurrentFrame(i);
        while (isBusy()) QApplication::processEvents();

        QImage tmp = m_output.convertToFormat(QImage::Format_ARGB32);
        encoder->append_frame(tmp.bits());

        qApp->processEvents();
        if (progress.wasCanceled())
            break;
    }

    encoder->finish();
    delete encoder;

    setCurrentFrame(0);
    settings.setValue("savename", filename);
    QDesktopServices::openUrl(QUrl::fromLocalFile(filename));
    #endif
}


void VideoPlayer::customEvent(QEvent *e) {
    #ifdef HAVE_LIBAV
    if (m_thread && (e->type() == QEvent::User)) {
        if (isBusy()) {
            QList<VideoPlayer*> L = slaves();
            for (int i = 0; i < L.size(); ++i) {
                if (L[i]->isBusy()) return;
            }

            {
                QMutexLocker lock(&m_thread->m_cacheMutex);
                if (m_thread->m_activeFrame < 0) return;
                m_currentFrame = m_thread->m_activeFrame;
                m_image = m_thread->m_cache.at(m_currentFrame);
            }

            currentFrameChanged(m_currentFrame);
            currentFrameChanged(m_image);

            VideoPlayer *p = qobject_cast<VideoPlayer*>(parent());
            if (p) {
                p->customEvent(e);
            }
        }
        return;
    }
    #endif
    QObject::customEvent(e);
}


void VideoPlayer::timerEvent(QTimerEvent *e) {
    #ifdef HAVE_LIBAV
    if (m_timer.timerId() == e->timerId()) {
        bool isActive = m_timer.isActive();
        m_timer.stop();

        if (m_thread) {
            libav_decoder *f = m_thread->m_libav;

            qint64 x = f->ticks() - m_ticks;
            qint64 t = f->time_from_ticks(x);
            
            int ms;
            if (t < f->duration()) {
                int frame = f->frame_from_time(t);
                qint64 t2 = f->time_from_frame(frame + 1);
                qint64 x2 = f->ticks_from_time(t2);
                ms = qMax<int>(10, (x2 - x) / 2000);
                setCurrentFrame(frame);
            } else {
                setCurrentFrame(0);
                m_ticks = f->ticks();
                m_ticks -= f->ticks_from_time(f->start_time());
                ms = 0;
            }
            
            if (isActive) {
                m_timer.start(ms, this);
            }
        }
        return;
    }
    #endif
    QObject::timerEvent(e);
}
