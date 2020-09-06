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
#include "videocontrols.h"


VideoControls::VideoControls(QWidget *parent) : QFrame (parent) {
    setEnabled(false);
    m_isTracking = false;
    m_isPlaying = false;
    m_autoHide = false;

    m_hbox = new QHBoxLayout(this);
    m_hbox->setSizeConstraint(QLayout::SetMinimumSize);
    m_hbox->setContentsMargins(0,0,0,0);
    m_hbox->setSpacing(4);
    
    m_prevButton = new QToolButton(this);
    m_hbox->addWidget(m_prevButton);
    m_prevButton->setFixedSize(24, 24);
    m_prevButton->setAutoRaise(true);
    m_prevButton->setIcon(QApplication::style()->standardIcon(QStyle::SP_MediaSeekBackward));

    m_playButton = new QToolButton(this);;
    m_hbox->addWidget(m_playButton);
    m_playButton->setFixedSize(24, 24);
    m_playButton->setAutoRaise(true);
    m_playButton->setIcon(QApplication::style()->standardIcon(QStyle::SP_MediaPlay));

    m_nextButton = new QToolButton(this);;
    m_hbox->addWidget(m_nextButton);
    m_nextButton->setFixedSize(24, 24);
    m_nextButton->setAutoRaise(true);
    m_nextButton->setIcon(QApplication::style()->standardIcon(QStyle::SP_MediaSeekForward));

    m_frameSlider = new QSlider(this);
    m_hbox->addWidget(m_frameSlider);
    m_frameSlider->setMinimumHeight(24);
    m_frameSlider->setRange(0,0);
    m_frameSlider->setFocusPolicy(Qt::NoFocus);
    m_frameSlider->setOrientation(Qt::Horizontal);
    m_frameSlider->setTracking(false);

    m_frameEdit = new QSpinBox(this);
    m_hbox->addWidget(m_frameEdit);
    m_frameEdit->setMinimumHeight(24);
    m_frameEdit->setRange(0,0);
    m_frameEdit->setButtonSymbols(QAbstractSpinBox::NoButtons);
    m_frameEdit->setAlignment(Qt::AlignCenter);
    m_frameEdit->setKeyboardTracking(false);

    connect(m_prevButton, SIGNAL(clicked()), this, SIGNAL(stepBack()));
    connect(m_playButton, SIGNAL(clicked()), this, SLOT(toogle()));
    connect(m_nextButton, SIGNAL(clicked()), this, SIGNAL(stepForward()));

    connect(m_frameSlider, SIGNAL(valueChanged(int)), this, SIGNAL(currentFrameChanged(int)));
    connect(m_frameSlider, SIGNAL(sliderMoved(int)), this, SIGNAL(currentFrameTracked(int)));
    connect(m_frameSlider, SIGNAL(sliderPressed()), this, SLOT(handleSliderPressed()));
    connect(m_frameSlider, SIGNAL(sliderReleased()), this, SLOT(handleSliderReleased()));

    connect(m_frameEdit, SIGNAL(valueChanged(int)), this, SIGNAL(currentFrameChanged(int)));
}


VideoControls::~VideoControls() {
}


void VideoControls::setAutoHide(bool hide) {
    if (m_autoHide != hide) {
        m_autoHide = hide;
        this->setVisible(!m_autoHide || (m_frameSlider->maximum() > 0));
    }
}


void VideoControls::setFrameCount(int nframes) {
    this->setEnabled(nframes > 1);
    m_frameSlider->setMaximum(nframes - 1);
    m_frameEdit->setMaximum(nframes - 1);
    this->setVisible(!m_autoHide || (nframes > 1));
}


void VideoControls::setCurrentFrame(int frame) {
    m_frameSlider->setValue(frame);
    m_frameEdit->setValue(frame);
}


void VideoControls::setPlayback(bool playing) {
    if (m_isPlaying != playing) {
        m_isPlaying = playing;
        if (m_isPlaying) {
            m_playButton->setIcon(QApplication::style()->standardIcon(QStyle::SP_MediaPause));
        } else {
            m_playButton->setIcon(QApplication::style()->standardIcon(QStyle::SP_MediaPlay));
        }
        m_nextButton->setEnabled(!m_isPlaying);
        m_prevButton->setEnabled(!m_isPlaying);
        m_frameSlider->setEnabled(!m_isPlaying);
        m_frameEdit->setEnabled(!m_isPlaying);
        playbackChanged(m_isPlaying);
        if (m_isPlaying) {
            playbackStarted();
        } else {
            playbackPaused();
        }
    }
}

    
void VideoControls::play() {
    setPlayback(true);
}


void VideoControls::pause() {
    setPlayback(false);
}


void VideoControls::toogle() {
    if (m_isPlaying) {
        pause();
    } else {
        play();
    }
}


void VideoControls::handleSliderPressed() {
    m_isTracking = true;
    trackingChanged(true);
    trackingStarted();
}

void VideoControls::handleSliderReleased() {
    m_isTracking = false;
    trackingChanged(false);
    trackingStopped();
}