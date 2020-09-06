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
#include "mainwindow.h"
#include "param.h"
#include "paramui.h"
#include "cudadevicedialog.h"
#include "imageutil.h"
#include "gpu_image.h"
#include "gpu_ivacef.h"
#include "gpu_color.h"
#include "gpu_st.h"
#include "gpu_stgauss3.h"
#include "gpu_gauss.h"
#include "gpu_util.h"


MainWindow::MainWindow() {
    setupUi(this);
    m_dirty = false;

    m_logWindow->setVisible(false);
    m_imageView->setFocus();
    m_imageView->setHandler(this);

    ParamGroup *g;
    new ParamInt(this, "N", 10, 1, 100, 1, &m_N);

    g = new ParamGroup(this, "structure tensor");
    new ParamDouble (g, "sigma_d", 1.0, 0.0, 10.0, 0.05, &m_sigma_d);
    new ParamDouble (g, "tau_r", 0.002, 0.0, 1.0, 0.001, &m_tau_r);
    new ParamInt    (g, "jacobi_steps", 1, 0, 1000, 1, &m_jacobi_steps);

    g = new ParamGroup(this, "smoothing");
    new ParamChoice (g, "order", "rk2", "euler|rk2", &m_order);
    new ParamDouble (g, "sigma_t", 6.0, 0.0, 20.0, 1, &m_sigma_t);
    new ParamDouble (g, "step_size", 1, 0.01, 10.0, 0.1, &m_step_size);
    new ParamBool   (g, "adaptive", true, &m_adaptive);
    new ParamChoice (g, "st_sampling", "linear", "nearest|linear", &m_st_sampling);
    new ParamChoice (g, "color_sampling", "linear", "nearest|linear", &m_color_sampling);

    g = new ParamGroup(this, "shock filtering", true, &m_shock_filtering);
    new ParamDouble (g, "sigma_i", 0.0, 0.0, 10.0, 0.25, &m_sigma_i);
    new ParamDouble (g, "sigma_g", 1.5, 0.0, 10.0, 0.25, &m_sigma_g);
    new ParamDouble (g, "r", 2, 0.0, 10.0, 0.25, &m_radius);
    new ParamDouble (g, "tau_s", 0.005, -2, 2, 0.01, &m_tau_s);

    g = new ParamGroup(this, "edge smoothing", true, &m_edge_smooting);
    new ParamDouble (g, "sigma_a", 1.5, 0.0, 10.0, 0.25, &m_sigma_a);

    g = new ParamGroup(this, "debug", false, &debug);
    new ParamBool   (g, "draw_orientation", false, &draw_orientation);
    new ParamBool   (g, "draw_streamline", false, &draw_streamline);

    ParamUI *pui = new ParamUI(this, this);
    pui->setFixedWidth(280);
    pui->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    m_vbox1->addWidget(pui);
    m_vbox1->addStretch(100);

    connect(m_select, SIGNAL(currentIndexChanged(int)), this, SLOT(onIndexChanged(int)));

    m_player = new VideoPlayer(this, ":/test.png");
    connect(m_player, SIGNAL(videoChanged(int)), this, SLOT(onVideoChanged(int)));
    connect(m_player, SIGNAL(currentFrameChanged(int)), this, SLOT(setDirty()));
    connect(m_player, SIGNAL(outputChanged(const QImage&)), m_imageView, SLOT(setImage(const QImage&)));
    connect(this, SIGNAL(imageChanged(const QImage&)), m_player, SLOT(setOutput(const QImage&)));

    m_videoControls->setFrameStyle(QFrame::NoFrame);
    m_videoControls->setAutoHide(true);
    connect(m_videoControls, SIGNAL(stepForward()), m_player, SLOT(stepForward()));
    connect(m_videoControls, SIGNAL(stepBack()), m_player, SLOT(stepBack()));
    connect(m_videoControls, SIGNAL(currentFrameTracked(int)), m_player, SLOT(setCurrentFrame(int)));
    connect(m_videoControls, SIGNAL(playbackChanged(bool)), m_player, SLOT(setPlayback(bool)));
    connect(m_videoControls, SIGNAL(trackingChanged(bool)), this, SLOT(setDirty()));

    connect(m_player, SIGNAL(videoChanged(int)), m_videoControls, SLOT(setFrameCount(int)));
    connect(m_player, SIGNAL(playbackChanged(bool)), m_videoControls, SLOT(setPlayback(bool)));
    connect(m_player, SIGNAL(currentFrameChanged(int)), m_videoControls, SLOT(setCurrentFrame(int)));
}


MainWindow::~MainWindow() {
}


void MainWindow::restoreAppState() {
    QSettings settings;
    restoreGeometry(settings.value("mainWindow/geometry").toByteArray());
    restoreState(settings.value("mainWindow/windowState").toByteArray());

    settings.beginGroup("imageView");
    m_imageView->restoreSettings(settings);
    settings.endGroup();

    settings.beginGroup("parameters");
    AbstractParam::restoreSettings(settings, this);
    settings.endGroup();

    m_player->restoreSettings(settings);
}


void MainWindow::saveAppState() {
    QSettings settings;
    settings.setValue("mainWindow/geometry", saveGeometry());
    settings.setValue("mainWindow/windowState", saveState());

    settings.beginGroup("imageView");
    m_imageView->saveSettings(settings);
    settings.endGroup();

    settings.beginGroup("parameters");
    AbstractParam::saveSettings(settings, this);
    settings.endGroup();

    m_player->saveSettings(settings);
}


bool MainWindow::event(QEvent *event) {
    if (event->type() == QEvent::Close) {
        saveAppState();
    }
    bool result = QMainWindow::event(event);
    if (event->type() == QEvent::Polish) {
        restoreAppState();
    }
    return result;
}


void MainWindow::on_actionOpen_triggered() {
    m_player->open();
}


void MainWindow::on_actionAbout_triggered() {
    QMessageBox msgBox;
    msgBox.setWindowTitle("About");
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setText(
        "<html><body>" \
        "<p><b>Coherence-Enhancing Filtering on the GPU</b><br/><br/>" \
        "Copyright (C) 2010-2012 Hasso-Plattner-Institut,<br/>" \
        "Fachgebiet Computergrafische Systeme &lt;" \
        "<a href='http://www.hpi3d.de'>www.hpi3d.de</a>&gt;<br/><br/>" \
        "Author: Jan Eric Kyprianidis &lt;" \
        "<a href='http://www.kyprianidis.com'>www.kyprianidis.com</a>&gt;<br/>" \
        "Date: " __DATE__ "</p>" \
        "<p>This program is free software: you can redistribute it and/or modify " \
        "it under the terms of the GNU General Public License as published by " \
        "the Free Software Foundation, either version 3 of the License, or " \
        "(at your option) any later version.</p>" \
        "<p>This program is distributed in the hope that it will be useful, " \
        "but WITHOUT ANY WARRANTY; without even the implied warranty of " \
        "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the " \
        "GNU General Public License for more details.</p>" \
        "Related Publications:" \
        "<ul>" \
        "<li>" \
        "Kyprianidis, J. E., &amp; Kang, H. (2011). " \
        "Image and Video Abstraction by Coherence-Enhancing Filtering. " \
        "<em>Computer Graphics Forum</em>, 30(2), 593-602. " \
        "(Proceedings Eurographics 2011)" \
        "</li>" \
        "<li>" \
        "Kyprianidis, J. E., &amp; Kang, H. (2013). " \
        "Coherence-Enhancing Filtering on the GPU. " \
        "<em>GPU Pro 4: Advanced Rendering Techniques</em>." \
        "</li>" \
        "</ul>" \
        "<p>Test image courtesy of Ivan Mlinaric @ flickr.com.</p>" \
        "</body></html>"
    );
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.exec();
}


void MainWindow::on_actionSelectDevice_triggered() {
    int current = 0;
    cudaGetDevice(&current);
    int N = CudaDeviceDialog::select(true);
    if ((N >= 0) && (current != N)) {
        QMessageBox::information(this, "Information", "Application must be restarted!");
        qApp->quit();
    }
}

void MainWindow::on_actionRecord_triggered() {
    m_player->record();
}


void MainWindow::on_actionLoadSettings_triggered() {
    QSettings settings;
    QString inputPath = window()->windowFilePath();
    QString outputPath = settings.value("savename", inputPath).toString();

    QString filename;
    QFileInfo fi(inputPath);
    QFileInfo fo(outputPath);
    if (!fi.baseName().isEmpty()) {
        QFileInfo fn(fo.dir(), fi.baseName() + ".ini");
        filename  = fn.absoluteFilePath();
    } else {
        filename  = fo.absolutePath();
    }

    filename = QFileDialog::getOpenFileName(this, "Load Settings", filename,
        "INI Format (*.ini);;All files (*.*)");
    if (!filename.isEmpty()) {
        QSettings iniFile(filename, QSettings::IniFormat);
        AbstractParam::restoreSettings(iniFile, this);
        settings.setValue("savename", filename);
    }
}


void MainWindow::on_actionSaveSettings_triggered() {
    QSettings settings;
    QString inputPath = window()->windowFilePath();
    QString outputPath = settings.value("savename", inputPath).toString();

    QString filename;
    QFileInfo fi(inputPath);
    QFileInfo fo(outputPath);
    if (!fi.baseName().isEmpty()) {
        QFileInfo fn(fo.dir(), fi.baseName() + ".ini");
        filename  = fn.absoluteFilePath();
    } else {
        filename  = fo.absolutePath();
    }

    filename = QFileDialog::getSaveFileName(this, "Save Settings", filename,
        "INI Format (*.ini);;All files (*.*)");
    if (!filename.isEmpty()) {
        QSettings iniFile(filename, QSettings::IniFormat);
        iniFile.clear();
        AbstractParam::saveSettings(iniFile, this);
        settings.setValue("savename", filename);
    }
}


void MainWindow::setDirty() {
    if (m_videoControls->isTracking()) {
        imageChanged(m_player->image());
    }
    else if (!m_dirty) {
        m_dirty = true;
        QMetaObject::invokeMethod(this, "process", Qt::QueuedConnection);
    }
}


void MainWindow::process() {
    m_dirty = false;
    QImage src = m_player->image();
    if (src.isNull()) {
        m_result[0] = m_result[1] = src;
        imageChanged(image());
        return;
    }

    gpu_image<float4> img = gpu_image_from_qimage<float4>(src);
    gpu_image<float4> st;

    for (int k = 0; k < m_N; ++k) {
        st = gpu_cef_st(img, st, m_sigma_d, m_tau_r, m_jacobi_steps);
        if (k == m_N-1) m_st = st.cpu();
        img = gpu_stgauss3_filter(
            img, 
            st, 
            m_sigma_t,
            m_color_sampling == "linear", 
            m_st_sampling == "linear", 
            (m_order == "rk2")? 2 : 1, 
            m_step_size, 
            m_adaptive);

        if (m_shock_filtering) {
            st = gpu_cef_st(img, st, m_sigma_d, m_tau_r, m_jacobi_steps);
            gpu_image<float> L = gpu_rgb2gray(img);
            L =  gpu_gauss_filter_xy(L, m_sigma_i);
            gpu_image<float> sign = gpu_cef_flog(L, st, m_sigma_g);
            img = gpu_cef_shock(L, st, sign, img, m_radius, m_tau_s);
        }
    }

    if (m_edge_smooting) {
        img = gpu_stgauss3_filter(img, st, m_sigma_a, true, true, 2, 1, false);
    }

    m_result[0] = src;
    m_result[1] = gpu_image_to_qimage(img);

    imageChanged(image());
}


void MainWindow::onIndexChanged(int index) {
    imageChanged(image());
}


void MainWindow::onVideoChanged(int nframes) {
    gpu_cache_clear();
    window()->setWindowFilePath(m_player->filename());
    window()->setWindowTitle(m_player->filename() + "[*] - cefabs");
    actionRecord->setEnabled(nframes > 1);
}


void MainWindow::draw(ImageView *view, QPainter &p, const QRectF& R, const QImage& image) {
    Handler::draw(view, p, R, image);
    if (debug) {
        if (draw_orientation && (m_imageView->scale() > 6)) {
            QRect aR = R.toAlignedRect();

            p.setPen(QPen(Qt::blue, 1 / m_imageView->scale()));
            for (int j = aR.top(); j <= aR.bottom(); ++j) {
                for (int i = aR.left(); i <= aR.right(); ++i) {
                    float2 t = normalize(st_minor_ev(m_st(i, j)));
                    QPointF q(i+0.5, j+0.5);
                    QPointF v(0.45 * t.x, 0.45 * t.y);

                    p.drawLine(q-v, q+v);
                }
            }
        }

        if (draw_streamline && m_st.is_valid() && (m_imageView->scale() > 6)) {
            QPointF c = QPointF(floor(R.center().x()) + 0.5f, floor(R.center().y()) + 0.5f);
            std::vector<float3> path = gpu_stgauss3_path(
                (int)c.x(), (int)c.y(), 
                m_st, 
                m_sigma_t, 
                m_st_sampling == "linear", 
                (m_order == "rk2")? 2 : 1, 
                m_step_size, 
                m_adaptive);

            QPolygonF P;
            for (int i = 0; i < (int)path.size(); ++i) {
                P.append(QPointF(path[i].x, path[i].y));
            }

            if (m_imageView->scale() > 30) {
                p.setPen(QPen(Qt::red, view->pt2px(10), Qt::SolidLine, Qt::RoundCap));
                p.drawPoint(c);
            }

            p.setPen(QPen(Qt::black, view->pt2px(2), Qt::SolidLine, Qt::RoundCap));
            p.drawPolyline(P);

            if (m_imageView->scale() > 30) {
                p.setPen(QPen(Qt::black, view->pt2px(5.0), Qt::SolidLine, Qt::RoundCap));
                p.drawPoints(P);
            }
        }
    }
}


void MainWindow::on_actionSavePNG_triggered() {
    m_imageView->savePNG(AbstractParam::paramText(this));
}
