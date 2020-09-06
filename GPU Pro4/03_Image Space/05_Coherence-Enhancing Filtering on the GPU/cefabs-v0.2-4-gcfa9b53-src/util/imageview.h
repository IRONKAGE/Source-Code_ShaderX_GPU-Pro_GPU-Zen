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

class ImageView : public QWidget {
    Q_OBJECT
    Q_PROPERTY(QImage image READ image WRITE setImage NOTIFY imageChanged(const QImage&)) 
    Q_PROPERTY(double zoom READ zoom WRITE setZoom NOTIFY zoomChanged(double)) 
    Q_PROPERTY(double resolution READ resolution WRITE setResolution NOTIFY resolutionChanged)
    Q_PROPERTY(double originX READ originX WRITE setOriginX NOTIFY originXChanged)
    Q_PROPERTY(double originY READ originY WRITE setOriginY NOTIFY originYChanged)
public:
    ImageView(QWidget *parent);
    ~ImageView();

    virtual void restoreSettings(QSettings& settings);
    virtual void saveSettings(QSettings& settings);

    QSize imageSize() const { return image().size(); }
    const QImage& image() const { return m_images[m_index]; }
    QPointF origin() const { return m_origin; }
    double originX() const { return m_origin.x(); }
    double originY() const { return m_origin.y(); }
    double zoom() const { return m_zoom; }
    double resolution() const { return m_resolution; }
    double scale() const { return m_zoom * m_resolution; }

    virtual QPointF view2image(const QPointF& p) const;
    virtual QPointF image2view(const QPointF& p) const;
    virtual QTransform viewTransform(const QSizeF& sz, double zoom, const QPointF& origin) const;
    virtual float pt2px(float pt) const;

    struct Handler {
        virtual void draw(ImageView *view, QPainter &p, const QRectF& R, const QImage& image);
        virtual void dragBegin(ImageView *view, QMouseEvent *e);
        virtual void dragMove(ImageView *view, QMouseEvent *e, const QPoint& start);
        virtual void dragEnd(ImageView *view, QMouseEvent *e, const QPoint& start);
    };
    void setHandler(Handler *handler);
    void setOverlay(QWidget *overlay);

public slots:
    void setImage(const QImage& image);
    void setOriginX(double value);
    void setOriginY(double value);
    void setZoom(double value);
    void setResolution(double value);
    void zoomIn();
    void zoomOut();
    void reset();
    void hold();
    void toggle();
    void copy();
    void savePNG(const QString& text=QString());

signals:
    void imageChanged(const QImage&);
    void originXChanged(double);
    void originYChanged(double);
    void zoomChanged(double);
    void resolutionChanged(double);

protected:
    virtual void paintEvent( QPaintEvent* );
    virtual void paint(QPainter &p, const QSizeF& sz, double zoom, const QPointF& origin);
    virtual void draw(QPainter& p, const QRectF& R, const QImage& image);

    virtual bool eventFilter( QObject *watched, QEvent *e );
    virtual void leaveEvent( QEvent *e );
    virtual void keyPressEvent( QKeyEvent *e );
    virtual void keyReleaseEvent( QKeyEvent *e );
    virtual void mousePressEvent( QMouseEvent *e );
    virtual void mouseMoveEvent( QMouseEvent *e );
    virtual void mouseReleaseEvent( QMouseEvent *e );
    virtual void wheelEvent(QWheelEvent *e);
    virtual void mouseDragBegin(QMouseEvent *e);
    virtual void mouseDragMove(QMouseEvent *e, const QPoint& start);
    virtual void mouseDragEnd(QMouseEvent *e, const QPoint& start);

protected:
    QPointF m_origin;
    double m_zoom;
    double m_resolution;
    bool m_spacePressed;
    
    enum Mode { MODE_NONE=0, MODE_PAN, MODE_DRAG };
    Mode m_mode;
    QPoint m_dragStart;
    Qt::MouseButton m_dragButton;
    QPointF m_dragOrigin;
    QCursor m_cursor;
    
    int m_index;
    QImage m_images[2];
    Handler *m_handler;
    QWidget *m_overlay;
};
