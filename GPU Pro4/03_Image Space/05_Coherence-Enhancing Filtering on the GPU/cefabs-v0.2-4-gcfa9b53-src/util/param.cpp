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
#include "param.h"


AbstractParam::AbstractParam( QObject *parent, const char* name, QVariant::Type type ) 
    : QObject(parent), m_type(type)
{
    setObjectName(name);

    QObject *p = parent;
    if (p) {
        const QMetaObject *meta = p->metaObject();
        if (meta->indexOfSlot("setDirty()") >= 0) {
            connect(this, SIGNAL(dirty()), p, SLOT(setDirty()));
        }
    }
}


const QVariantHash AbstractParam::attributes() const {
    QVariantHash a;
    const QMetaObject *mo = metaObject();
    for (int k = AbstractParam::staticMetaObject.propertyOffset(); k < mo->propertyCount(); ++k) {
        const QMetaProperty mp = mo->property(k);
        a.insert(mp.name(), this->property(mp.name()));
    }
    return a;
}


QString AbstractParam::tooltip() const {
    QStringList tooltip;
    const QMetaObject *mo = metaObject();
    for (int k = AbstractParam::staticMetaObject.propertyOffset(); k < mo->propertyCount(); ++k) {
        const QMetaProperty mp = mo->property(k);
        QVariant v = this->property(mp.name());
        QString s;
        if (v.type() == QVariant::StringList) {
            s = v.toStringList().join(",");
        } else {
            s = v.toString();
        }
        tooltip.append(QString("%1=%2").arg(mp.name()).arg(s));
    }
    return tooltip.join("\n");
}


void AbstractParam::saveValue(QSettings& settings) {
    if ((type() != QVariant::Invalid) && !name().isEmpty()) {
        settings.setValue(name(), value());
    }
}


void AbstractParam::restoreValue(QSettings& settings) {
    if ((type() != QVariant::Invalid) && !name().isEmpty()) {
        QVariant v = settings.value(name());
        setValue(v);
    }
}


void AbstractParam::saveSettings(QSettings& settings, QObject *object) {
    const QObjectList& L = object->children();
    for (int i = 0; i < L.size(); ++i) {
        AbstractParam *p = qobject_cast<AbstractParam*>(L[i]);
        if (p) {
            p->saveValue(settings);
            if (!p->children().isEmpty()) {
                settings.beginGroup(p->name());
                saveSettings(settings, p);
                settings.endGroup();
            }
        }
    }
}


void AbstractParam::restoreSettings(QSettings& settings, QObject *object) {
    const QObjectList& L = object->children();
    for (int i = 0; i < L.size(); ++i) {
        AbstractParam *p = qobject_cast<AbstractParam*>(L[i]);
        if (p) {
            p->restoreValue(settings);
            if (!p->children().isEmpty()) {
                settings.beginGroup(p->name());
                restoreSettings(settings, p);
                settings.endGroup();
            }
        }
    }
}


void AbstractParam::setDirty() {
    if (type() != QVariant::Invalid) {
        valueChanged(value());
    }
    dirty();
}


void AbstractParam::resetValue() {
    const QMetaObject *mo = metaObject();
    int i = mo->indexOfProperty("value");
    if (i >= 0) {
        QMetaProperty mp = mo->property(i);
        if (mp.isResettable()) {
            mp.reset(this);
        }
    }
}


void AbstractParam::setValue(const QVariant& value) {
    setProperty("value", value);
}


QString AbstractParam::paramText(QObject *object) {
    const QObjectList& L = object->children();
    QString text;
    for (int i = 0; i < L.size(); ++i) {
        AbstractParam *p = qobject_cast<AbstractParam*>(L[i]);
        if (p) {
            text += QString("%1=%2; ").arg( p->name()).arg(p->value().toString());
            if (!p->children().isEmpty()) {
                text += AbstractParam::paramText(p);
            }
        }
    }
    return text;
}


ParamGroup::ParamGroup( QObject *parent, const char* name ) 
    : AbstractParam(parent, name, QVariant::Invalid)
{
    m_defaultValue = true;
    m_ptr = 0;
    m_value = true;
}


ParamGroup::ParamGroup( QObject *parent, const char* name, bool value, bool *ptr ) 
    : AbstractParam(parent, name, QVariant::Bool)
{
    m_defaultValue = value;
    m_ptr = ptr;
    m_value = value;
    if (m_ptr) *m_ptr = m_value;
}


void ParamGroup::setDirty() {
    if (type() == QVariant::Bool) {
        valueChanged(m_value);
    }
    AbstractParam::setDirty();
}


void ParamGroup::setValue(bool value) {
    if ((type() == QVariant::Bool) && (value != m_value)) {
        m_value = value;
        if (m_ptr) *m_ptr = value;
        setDirty();
    }
}


ParamBool::ParamBool( QObject *parent, const char* name, bool value, bool *ptr ) 
    : AbstractParam(parent, name, QVariant::Bool)
{
    m_defaultValue = value;
    m_ptr = ptr;
    m_value = value;
    if (m_ptr) *m_ptr = m_value;
}


void ParamBool::setDirty() {
    valueChanged(m_value);
    AbstractParam::setDirty();
}


void ParamBool::setValue(bool value) {
    if (value != m_value) {
        m_value = value;
        if (m_ptr) *m_ptr = value;
        setDirty();
    }
}


ParamInt::ParamInt( QObject *parent, const char* name, int value, int minimum, int maximum, 
                    int singleStep, int *ptr ) : AbstractParam(parent, name, QVariant::Int)
{
    m_defaultValue = value;
    m_minimum = minimum;
    m_maximum = maximum;
    m_singleStep = singleStep;
    m_ptr = ptr;
    m_value = qBound(minimum, m_defaultValue, maximum);
    if (m_ptr) *m_ptr = m_value;
}


void ParamInt::setRange(int minimum, int maximum) {
    m_minimum = minimum;
    m_maximum = maximum;
    setValue(m_value);
}


void ParamInt::setDirty() {
    valueChanged(m_value);
    AbstractParam::setDirty();
}


void ParamInt::setValue(int value) {
    int v = qBound(m_minimum, value, m_maximum);
    if (v != m_value) {
        m_value = v;
        if (m_ptr) *m_ptr = v;
        setDirty();
    }
}


ParamDouble::ParamDouble( QObject *parent, const char* name, double value, double minimum, double maximum, 
                          double singleStep, double *ptr ) : AbstractParam(parent, name, QVariant::Double)
{
    m_defaultValue = value;
    m_minimum = minimum;
    m_maximum = maximum;
    m_singleStep = singleStep;
    m_decimals = 5;
    m_ptr = ptr;
    m_value = qBound(minimum, m_defaultValue, maximum);
    if (m_ptr) *m_ptr = m_value;
}


void ParamDouble::setRange(double minimum, double maximum) {
    m_minimum = minimum;
    m_maximum = maximum;
    setValue(m_value);
}


void ParamDouble::setDirty() {
    valueChanged(m_value);
    AbstractParam::setDirty();
}


void ParamDouble::setValue(double value) {
    double v = qBound(m_minimum, value, m_maximum);
    if (fabs(v - m_value) > 1e-8) {
        m_value = v;
        if (m_ptr) *m_ptr = v;
        setDirty();
    }
}


ParamString::ParamString( QObject *parent, const char* name, const QString& value, 
                          bool multiLine, QString *ptr) : AbstractParam(parent, name, QVariant::String)
{
    m_defaultValue = value;
    m_multiLine = multiLine;
    m_ptr = ptr;
    m_value = value;
    if (m_ptr) *m_ptr = m_value;
}


void ParamString::setDirty() {
    valueChanged(m_value);
    AbstractParam::setDirty();
}


void ParamString::setValue(const QString& value) {
    if (value != m_value) {
        m_value = value;
        if (m_ptr) *m_ptr = value;
        setDirty();
    }
}


ParamChoice::ParamChoice( QObject *parent, const char* name, const QString& value, const QString& items, 
                          QString *ptr) : AbstractParam(parent, name, QVariant::String)
{
    m_defaultValue = value;
    QStringList L = items.split('|', QString::SkipEmptyParts);
    for (int i = 0; i < L.size(); ++i) {
        m_keys.append(L[i]);
        m_values.append(L[i]);
    }
    m_ptr = ptr;
    m_value = L.value(L.indexOf(value));
    if (m_ptr) *m_ptr = m_value;
    
}


void ParamChoice::setDirty() {
    valueChanged(m_value);
    AbstractParam::setDirty();
}


void ParamChoice::setValue(const QString& value) {
    if (value != m_value) {
        if (m_values.contains(value)) {
            m_value = value;
            if (m_ptr) *m_ptr = value;
            setDirty();
        }
    }
}


ParamEnum::ParamEnum( QObject *parent, const char* name, int value, const QString& e, int *ptr ) 
    : AbstractParam(parent, name, QVariant::Int)
{
    m_ptr = ptr;

    QStringList L = e.split('|', QString::SkipEmptyParts);
    if (!L.isEmpty()) {
        for (int i = 0; i < L.size(); ++i) {
            m_keys.append(L[i]);
            m_values.append(i);
        }
    } else {
        QObject *p = parent;
        while (p) {
            const QMetaObject *mo = p->metaObject();
            int index = mo->indexOfEnumerator(e.toLatin1().data());
            if (index >= 0) {
                const QMetaEnum me = mo->enumerator(index);
                for (int i = 0; i < me.keyCount(); ++i) {
                    m_keys.append(me.key(i));
                    m_values.append(me.value(i));
                }
                break;
            }
            p = p->parent();
        }
    }

    m_value = m_defaultValue = value;
    if (m_ptr) *m_ptr = m_value;
}


void ParamEnum::setDirty() {
    valueChanged(m_value);
    AbstractParam::setDirty();
}


void ParamEnum::setValue(const QVariant& value) {
    if (value.type() == QVariant::String) {
        int index = m_keys.indexOf(value.toString());
        if (index >= 0) {
            setValue(m_values[index]);
        }
    } else {
        QVariant v(value);
        if (v.convert(QVariant::Int)) {
            setValue(v.toInt());    
        }
    }
}


void ParamEnum::setValue(int value) {
    if (value != m_value) {
        if (m_values.indexOf(value) >= 0) {
            m_value = value;
            if (m_ptr) *m_ptr = value;
            setDirty();
        }
    }
}


ParamImage::ParamImage( QObject *parent, const char* name, const QImage& value, QImage*ptr) 
    : AbstractParam(parent, name, QVariant::Image)
{
    m_defaultValue = value;
    m_ptr = ptr;
    m_value = value;
    if (m_ptr) *m_ptr = m_value;
}


ParamImage::ParamImage( QObject *parent, const char* name, const QString& path, QImage*ptr) 
    : AbstractParam(parent, name, QVariant::Image)
{
    m_value = QImage(path);
    if (!m_value.isNull()) {
        m_value.setText("filename", path);    
    }
    m_defaultValue = m_value;
    m_ptr = ptr;
    if (m_ptr) *m_ptr = m_value;
}


void ParamImage::saveValue(QSettings& settings) {
    QString path = m_value.text("filename");
    QFileInfo fi(path);
    if (!fi.exists()) {
        path = "";
    }
    settings.setValue(name(), path);        
}


void ParamImage::restoreValue(QSettings& settings) {
    m_value = m_defaultValue;
    QString path = settings.value(name()).toString();
    if (path.isEmpty()) {
        path = m_defaultValue.text("filename");
    }
    QFileInfo fi(path);
    if (fi.exists()) {
        QImage image(path);
        if (!image.isNull()) {
            image.setText("filename", path);    
            setValue(image);
        }
    }
}


void ParamImage::setDirty() {
    valueChanged(m_value);
    AbstractParam::setDirty();
}


void ParamImage::setValue(const QImage& value) {
    if (value != m_value) {
        m_value = value;
        if (m_ptr) *m_ptr = value;
        setDirty();
    }
}
