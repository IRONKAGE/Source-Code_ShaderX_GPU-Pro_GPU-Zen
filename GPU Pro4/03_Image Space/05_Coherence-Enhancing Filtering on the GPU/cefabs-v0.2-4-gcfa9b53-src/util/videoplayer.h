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

class VideoPlayer : public QObject {
    Q_OBJECT
public:
    VideoPlayer( QObject *parent, const QString& filename=QString::null, 
                 int cacheRadius=0, int cacheMin=32, int cacheMax=128 );
    virtual ~VideoPlayer();
    
    void saveSettings(QSettings& settings);
    void restoreSettings(QSettings& settings);

    bool open(const QString& path);
    void record(QObject *r);

    const QString& filename() const;
    bool isValid() const;
    bool isBusy() const;
    QSize size() const;
    int width() const;
    int height() const;
    int frameCount() const;
    double fps() const;
    QPair<int,int> frameRate() const;
    QPair<int,int> timeBase() const;
    int currentFrame() const;
    QImage image(int index=0) const;
    qint64 time(int index=0) const;
    bool isPlaying() const;
    
    QList<VideoPlayer*> slaves() const;
    VideoPlayer* slave(int index) const;

public slots:
    void open();
    void close();
    void rewind();
    void stepForward();
    void stepBack();
    void setCurrentFrame(int frame);
    void setPlayback(bool playing);
    void play();
    void pause();
    void toggle();
    void setOutput(const QImage& image);
    void record();

signals:
    void videoChanged(const QSize& size);
    void videoChanged(int nframes);
    void currentFrameChanged(int frame);
    void currentFrameChanged(const QImage& image);
    void playbackChanged(bool playing);
    void playbackStarted();
    void playbackPaused();
    void outputChanged(const QImage& image);

protected:
    void customEvent(QEvent *e);
    void timerEvent(QTimerEvent *e);

    QString m_filename;
    QBasicTimer m_timer;
    qint64 m_ticks;
    int m_currentFrame;
    QImage m_image;
    QImage m_output;
    class Thread;
    Thread *m_thread;
    int m_cacheRadius;
    int m_cacheMin;
    int m_cacheMax;
};
