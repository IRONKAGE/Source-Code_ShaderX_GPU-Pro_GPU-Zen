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
#include "imageview.h"


ImageView::ImageView(QWidget *parent) : QWidget(parent) {
    m_spacePressed = false;
    m_mode = MODE_NONE;
    m_index = 0;
    m_handler = 0;
    m_overlay = 0;
    m_zoom = m_resolution = 1;

    setAutoFillBackground(true);
    QPalette p = palette();
    p.setBrush(QPalette::Background, p.mid());
    setPalette(p);

    qApp->installEventFilter(this);
}


ImageView::~ImageView() {
}


void ImageView::restoreSettings(QSettings& settings) {
    setZoom(settings.value("zoom", 1.0).toDouble());
    setResolution(settings.value("resolution", 1.0).toDouble());
    setOriginX(settings.value("originX", 0).toFloat());
    setOriginY(settings.value("originY", 0).toFloat());
}


void ImageView::saveSettings(QSettings& settings) {
    settings.setValue("zoom", m_zoom);
    settings.setValue("resolution", m_resolution);
    settings.setValue("originX", m_origin.x());
    settings.setValue("originY", m_origin.y());
}


QPointF ImageView::view2image(const QPointF& p) const {
    QSize sz = size();
    float z = m_zoom * m_resolution;
    return QPointF(
        (p.x() - m_origin.x() * z - sz.width() / 2) / z + imageSize().width() / 2,
        (p.y() - m_origin.y() * z - sz.height() / 2) / z + imageSize().height() / 2
    );
}


QPointF ImageView::image2view(const QPointF& q) const {
    QSize sz = size();
    float z = m_zoom * m_resolution;
    return QPointF(
        (q.x() - imageSize().width() / 2) * z + m_origin.x() * z + sz.width() / 2,
        (q.y() - imageSize().height() / 2) * z + m_origin.y() * z + sz.height() / 2
    );
}


QTransform ImageView::viewTransform(const QSizeF& sz, double zoom, const QPointF& origin) const {
    QTransform tr;
    tr.translate(sz.width()/2.0f, sz.height()/2.0f);
    tr.translate(origin.x() * zoom, origin.y() * zoom);
    tr.scale(zoom, zoom);
    tr.translate(-imageSize().width()/2.0f, -imageSize().height()/2.0f);
    return tr;
}


float ImageView::pt2px(float pt) const {
    return pt / (m_zoom * m_resolution);
}


void ImageView::setHandler(Handler *handler) {
    m_handler = handler;
    update();
}


void ImageView::setOverlay(QWidget *overlay) {
    if (overlay) {
        setMouseTracking(true);
        m_overlay = overlay;
        m_overlay->setVisible(false);
    } else {
        setMouseTracking(false);
        if (m_overlay) m_overlay->setVisible(false);
        m_overlay = 0;
    }
}


void ImageView::setImage(const QImage& image) {
    m_images[0] = image;
    imageChanged(m_images[0]);
    update();
}


void ImageView::setOriginX(double value) {
    double x = value;
    if (m_origin.x() != x) {
        m_origin.setX(x);
        originXChanged(x);
        update();
    }
}


void ImageView::setOriginY(double value) {
    double y = value;
    if (m_origin.y() != y) {
        m_origin.setY(y);
        originYChanged(y);
        update();
    }
}


void ImageView::setZoom(double value) {
    if (value != m_zoom) {
        m_zoom = value;
        zoomChanged(m_zoom);
        update();
    }
}


void ImageView::setResolution(double value) {
    if (value != m_resolution) {
        m_resolution = value;
        resolutionChanged(m_resolution);
        update();
    }
}


void ImageView::zoomIn() {
    setZoom(m_zoom * 2);
}


void ImageView::zoomOut() {
    setZoom(m_zoom / 2);
}


void ImageView::reset() {
    m_zoom = 1.0;
    m_resolution = 1.0;
    m_origin = QPoint(0, 0);
    zoomChanged(1);
    resolutionChanged(1);
    originXChanged(0);
    originYChanged(0);
    update();
}


void ImageView::hold() {
    m_images[1] = m_images[0];
    update();
}


void ImageView::toggle() {
    m_index = 1 - m_index;
    QString title = window()->windowTitle().replace(" -- hold", "");
    if (m_index)
        title += " -- hold";
    window()->setWindowTitle(title);
    update();
}


void ImageView::copy() {
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setImage(image());
}


void ImageView::savePNG(const QString& text) {
    QSettings settings;
    QString inputPath = window()->windowFilePath();
    QString outputPath = settings.value("savename", inputPath).toString();

    QString filename;
    QFileInfo fi(inputPath);
    QFileInfo fo(outputPath);
    if (!fi.baseName().isEmpty()) {
        QFileInfo fn(fo.dir(), fi.baseName() + "-out.png");
        filename  = fn.absoluteFilePath();
    } else {
        filename  = fo.absolutePath();
    }

    filename = QFileDialog::getSaveFileName(this, "Save PNG", filename, 
        "PNG Format (*.png);;All files (*.*)");
    if (!filename.isEmpty()) {
        QImage tmp(image());
        if (!text.isEmpty()) tmp.setText("Description", text);
        if (!tmp.save(filename)) {
            QMessageBox::critical(this, "Error", QString("Saving PNG '%1' failed!").arg(filename));
            return;
        }
        settings.setValue("savename", filename);
    }
}


void ImageView::paintEvent(QPaintEvent *e) {
    QSize sz = size();
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setWindow(0, 0, sz.width(), sz.height());
    p.scale(m_resolution, m_resolution);
    paint(p, sz, m_zoom, m_origin);
}


void ImageView::paint(QPainter &p, const QSizeF& sz, double zoom, const QPointF& origin) {
    if (!imageSize().isEmpty()) {
        QTransform tr = viewTransform(sz, zoom, origin);
        QTransform invTr = tr.inverted();
        QRectF wR = invTr.mapRect(QRectF(QPoint(0,0), sz));
        QRectF R = wR.intersected(QRectF(QPoint(0,0), imageSize()));
        if (!R.isEmpty()) {
            p.save();
            p.setTransform(tr, true);
            if (m_handler) {
                m_handler->draw(this, p, R, image());
            } else {
                draw(p, R, image());
            }
            p.restore();
        }
    }
}


void ImageView::draw(QPainter& p, const QRectF& R, const QImage& image) {
    QRect aR = R.toAlignedRect();
    p.drawImage(aR.x(), aR.y(), image.copy(aR));
}


bool ImageView::eventFilter( QObject *watched, QEvent *e ) {
    if (!hasFocus() && (m_mode == MODE_NONE) && (e->type() == QEvent::KeyPress)) {
        QKeyEvent *k = (QKeyEvent*)e;
        if (k->key() == Qt::Key_Space) {
            QPoint gp = mapFromGlobal(QCursor::pos());
            if (rect().contains(gp)) {
                setFocus(Qt::OtherFocusReason);
                keyPressEvent(k);
                return true;
            }
        }
    }
    return QWidget::eventFilter(watched, e);
}


void ImageView::leaveEvent( QEvent *e ) {
    if (m_spacePressed) {
        setCursor(m_cursor);
        m_cursor = QCursor();
        m_spacePressed = false;
    }
    if (m_overlay) {
        m_overlay->setVisible(false);
    }
    QWidget::leaveEvent(e);
}


void ImageView::keyPressEvent( QKeyEvent *e ) {
    if ((m_mode == MODE_NONE) && (e->key() == Qt::Key_Space)) {
        if (!m_spacePressed) {
            if (m_overlay && m_overlay->isVisible()) m_overlay->setVisible(false);
            m_cursor = cursor();
            setCursor(Qt::OpenHandCursor);
            m_spacePressed = true;
        }
    } else {
        QWidget::keyPressEvent(e);
    }
}


void ImageView::keyReleaseEvent( QKeyEvent *e ) {
    if (e->key() == Qt::Key_Space) {
        if (!e->isAutoRepeat()) {
            if (m_mode == MODE_NONE) {
                setCursor(m_cursor);
                m_cursor = QCursor();
            }
            m_spacePressed = false;
        }
    } else {
        QWidget::keyReleaseEvent(e);
    }
}


void ImageView::mousePressEvent( QMouseEvent *e ) {
    if ((m_mode == MODE_NONE) && (e->button() != Qt::NoButton)) {
        if (!m_spacePressed || (e->buttons() != Qt::LeftButton)) {
            m_mode = MODE_DRAG;
            m_dragStart =  e->pos();
            m_dragButton = e->button();
            mouseDragBegin(e);
            if (e->isAccepted()) return;
        } 
        setCursor(Qt::ClosedHandCursor);
        m_mode = MODE_PAN;
        m_dragStart = e->pos();
        m_dragButton = e->button();
        m_dragOrigin = m_origin;
        update();
    }
}


void ImageView::mouseMoveEvent( QMouseEvent *e ) {
    if (m_mode == MODE_PAN) {
        m_origin = m_dragOrigin + QPointF(e->pos() - m_dragStart) / m_zoom / m_resolution;
        originXChanged(m_origin.x());
        originYChanged(m_origin.y());
        update();
    } else if (m_mode == MODE_DRAG) {
        mouseDragMove(e, m_dragStart);
    }
    if (m_overlay && !m_spacePressed && (m_mode != MODE_PAN)) {
        QSize sz = m_overlay->size();
        m_overlay->move(e->pos().x() - sz.width()/2, e->pos().y() - sz.height()/2);
        if (m_overlay->isHidden()) m_overlay->setVisible(true);
    }
}


void ImageView::mouseReleaseEvent( QMouseEvent *e ) {
    if ((m_mode == MODE_PAN) && (e->button() == m_dragButton)) {
        if (m_spacePressed) {
            setCursor(QCursor(Qt::OpenHandCursor));
        } else {
            setCursor(m_cursor);
            m_cursor = QCursor();
        }
        m_mode = MODE_NONE;
        update();
    } else if ((m_mode == MODE_DRAG) && (e->button() == m_dragButton)) {
        m_mode = MODE_NONE;
        mouseDragEnd(e, m_dragStart);
    }
}


void ImageView::wheelEvent(QWheelEvent *e) {
    QSize sz = size();
    double u = e->delta() / 120.0 / 4.0;
    if (u < -0.5) u = -0.5;
    if (u > 0.5) u = 0.5;
    if (e->modifiers() & Qt::ControlModifier) {
        setResolution(m_resolution * (1 + u));
    } else {
        setZoom(m_zoom * (1 + u));
    }
}


void ImageView::mouseDragBegin(QMouseEvent *e) {
    if (m_handler) 
        m_handler->dragBegin(this, e);
    else
        e->ignore();
}


void ImageView::mouseDragMove(QMouseEvent *e, const QPoint& start) {
    if (m_handler) 
        m_handler->dragMove(this, e, start);
    else
        e->ignore();
}


void ImageView::mouseDragEnd(QMouseEvent *e, const QPoint& start) {
    if (m_handler) 
        m_handler->dragEnd(this, e, start);
    else
        e->ignore();
}


void ImageView::Handler::draw(ImageView *view, QPainter &p, const QRectF& R, const QImage& image) {
    view->draw(p, R, image);
}


void ImageView::Handler::dragBegin(ImageView *view, QMouseEvent *e) {
    e->ignore();
}


void ImageView::Handler::dragMove(ImageView *view, QMouseEvent *e, const QPoint& start) {
    e->ignore();
}


void ImageView::Handler::dragEnd(ImageView *view, QMouseEvent *e, const QPoint& start) {
    e->ignore();
}
