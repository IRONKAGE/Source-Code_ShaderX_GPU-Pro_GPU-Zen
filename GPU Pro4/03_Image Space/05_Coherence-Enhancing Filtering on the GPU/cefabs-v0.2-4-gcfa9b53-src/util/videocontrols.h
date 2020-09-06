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

class VideoControls : public QFrame {
    Q_OBJECT
public:
    VideoControls(QWidget *parent);
    virtual ~VideoControls();

    void setAutoHide(bool hide);
    bool autoHide() const { return m_autoHide; }
    bool isPlaying() const { return m_isPlaying; }
    bool isTracking() const { return m_isTracking; }

public slots:
    void setFrameCount(int nframes);
    void setCurrentFrame(int frame);
    void setPlayback(bool playing);
    void play();
    void pause();
    void toogle();

signals:
    void stepForward();
    void stepBack();
    void currentFrameChanged(int frame);
    void currentFrameTracked(int frame);
    void playbackChanged(bool playing);
    void playbackStarted();
    void playbackPaused();
    void trackingChanged(bool tracking);
    void trackingStarted();
    void trackingStopped();

protected slots:
    void handleSliderPressed();
    void handleSliderReleased();

protected:
    QHBoxLayout *m_hbox;
    bool m_autoHide;
    bool m_isPlaying;
    bool m_isTracking;
    QToolButton *m_prevButton;
    QToolButton *m_playButton;
    QToolButton *m_nextButton;
    QSlider *m_frameSlider;
    QSpinBox *m_frameEdit;
};
