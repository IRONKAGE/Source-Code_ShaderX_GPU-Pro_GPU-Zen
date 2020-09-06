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

#include "rolloutbox.h"

class AbstractParam;

class ParamUI : public RolloutBox {
    Q_OBJECT
public:
    ParamUI(QWidget *parent, QObject *object);

protected:
    void addObjectParameters(QObject *obj);
    QPointer<QObject> m_object;
};


class ParamLabel : public QLabel {
    Q_OBJECT
public:            
    ParamLabel(QWidget *parent, AbstractParam *param);

protected:
    void mouseDoubleClickEvent(QMouseEvent * e);
    QPointer<AbstractParam> m_param;
};


class ParamCheckBox : public QCheckBox {
    Q_OBJECT
public:            
    ParamCheckBox(QWidget *parent, AbstractParam *param);
    virtual bool event(QEvent *e);

protected:
    QPointer<AbstractParam> m_param;
};


class ParamSpinBox : public QSpinBox {
    Q_OBJECT
public:            
    ParamSpinBox(QWidget *parent, AbstractParam *param);

protected:
    QPointer<AbstractParam> m_param;
};


class ParamDoubleSpinBox : public QDoubleSpinBox {
    Q_OBJECT
public:            
    ParamDoubleSpinBox(QWidget *parent, AbstractParam *param);

protected:
    QPointer<AbstractParam> m_param;
};


class ParamComboBox : public QComboBox {
    Q_OBJECT
public:            
    ParamComboBox(QWidget *parent, AbstractParam *param);

protected slots:
    void setValue(const QVariant& value);
    void updateParam(int index);

protected:
    QPointer<AbstractParam> m_param;
};


class ParamLineEdit : public QLineEdit {
    Q_OBJECT
public:            
    ParamLineEdit(QWidget *parent, AbstractParam *param);

protected slots:
    void updateParam();

protected:
    QPointer<AbstractParam> m_param;
};


class ParamTextEdit : public QToolButton {
    Q_OBJECT
public:            
    ParamTextEdit(QWidget *parent, AbstractParam *param);

protected slots:
    void edit();

protected:
    QPointer<AbstractParam> m_param;
};


class ParamImageSelect : public QWidget {
    Q_OBJECT
public:            
    ParamImageSelect(QWidget *parent, AbstractParam *param);

protected slots:
    void setImage(const QImage& image);
    void clear();
    void edit();

protected:
    QPointer<AbstractParam> m_param;
    QString m_filename;
    QImage m_image;
    QLabel *m_label;
    QToolButton *m_edit;
    QToolButton *m_clear;
};
