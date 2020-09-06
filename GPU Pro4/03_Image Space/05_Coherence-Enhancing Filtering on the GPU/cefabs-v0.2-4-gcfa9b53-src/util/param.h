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

class AbstractParam : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString name READ name)
public:
    AbstractParam( QObject *parent, const char* name, QVariant::Type type );

    const QString name() const { return objectName(); }
    QVariant::Type type() const { return m_type; }
    QVariant value() const { return property("value"); }
    const QVariantHash attributes() const;
    QString tooltip() const;

    virtual void saveValue(QSettings& settings);
    virtual void restoreValue(QSettings& settings);

signals:
    void dirty();
    void valueChanged(const QVariant&);

public slots:
    virtual void resetValue();
    virtual void setDirty();
    virtual void setValue(const QVariant&);

protected:
    QVariant::Type m_type;

public:
    static void saveSettings(QSettings& settings, QObject *object);
    static void restoreSettings(QSettings& settings, QObject *object);
    static QString paramText(QObject *object);
};


class ParamGroup : public AbstractParam {
    Q_OBJECT
    Q_PROPERTY(bool value READ value WRITE setValue NOTIFY valueChanged USER true)
    Q_PROPERTY(bool defaultValue READ defaultValue )
public:
    ParamGroup( QObject *parent, const char* name );
    ParamGroup( QObject *parent, const char* name, bool value, bool *ptr=0 );

    bool value() const { return m_value; }
    bool defaultValue() const { return m_defaultValue; }

signals:
    void valueChanged(bool);

public slots:
    void setDirty();
    void setValue(bool);

protected:
    bool m_value;
    bool m_defaultValue;
    bool *m_ptr;
};


class ParamBool : public AbstractParam {
    Q_OBJECT
    Q_PROPERTY(bool value READ value WRITE setValue RESET reset NOTIFY valueChanged USER true)
public:
    ParamBool( QObject *parent, const char* name, bool value, bool *ptr=0 );

    bool value() const { return m_value; }
    void reset() { setValue(m_defaultValue); }

signals:
    void valueChanged(bool);

public slots:
    void setDirty();
    void setValue(bool);

protected:
    bool m_value;
    bool m_defaultValue;
    bool *m_ptr;
};


class ParamInt : public AbstractParam {
    Q_OBJECT
    Q_PROPERTY(int value READ value WRITE setValue RESET reset NOTIFY valueChanged USER true)
    Q_PROPERTY(int minimum READ minimum WRITE setMinimum)
    Q_PROPERTY(int maximum READ maximum WRITE setMaximum)
    Q_PROPERTY(int singleStep READ singleStep WRITE setSingleStep)
public:
    ParamInt( QObject *parent, const char* name, int value, int minimum, 
              int maximum, int singleStep, int *ptr=0 );

    int value() const { return m_value; }
    void reset() { setValue(m_defaultValue); }
    int minimum() const { return m_minimum; }
    int maximum() const { return m_maximum; }
    int singleStep() const { return m_singleStep; }

    void setRange(int minimum, int maximum);
    void setMinimum(int minimum) { setRange(minimum, m_maximum); }
    void setMaximum(int maximum) { setRange(m_minimum, maximum); }
    void setSingleStep(int singleStep) { m_singleStep = singleStep; }

signals:
    void valueChanged(int);

public slots:
    void setDirty();
    void setValue(int);

protected:
    int m_value;
    int m_defaultValue;
    int m_minimum;
    int m_maximum;
    int m_singleStep;
    int *m_ptr;
};


class ParamDouble : public AbstractParam {
    Q_OBJECT
    Q_PROPERTY(double value READ value WRITE setValue RESET reset NOTIFY valueChanged USER true)
    Q_PROPERTY(double minimum READ minimum WRITE setMinimum)
    Q_PROPERTY(double maximum READ maximum WRITE setMaximum)
    Q_PROPERTY(double singleStep READ singleStep WRITE setSingleStep)
    Q_PROPERTY(int decimals READ decimals WRITE setDecimals)
public:
    ParamDouble( QObject *parent, const char* name, double value, double minimum, 
                 double maximum, double singleStep, double *ptr=0 );

    double value() const { return m_value; }
    void reset() { setValue(m_defaultValue); }
    double minimum() const { return m_minimum; }
    double maximum() const { return m_maximum; }
    double singleStep() const { return m_singleStep; }
    int decimals() const { return m_decimals; }

    void setRange(double minimum, double maximum);
    void setMinimum(double minimum) { setRange(minimum, m_maximum); }
    void setMaximum(double maximum) { setRange(m_minimum, maximum); }
    void setSingleStep(double singleStep) { m_singleStep = singleStep; }
    void setDecimals(int decimals) { m_decimals = decimals; }

signals:
    void valueChanged(double);

public slots:
    void setDirty();
    void setValue(double);

protected:
    double m_value;
    double m_defaultValue;
    double m_minimum;
    double m_maximum;
    double m_singleStep;
    int m_decimals;
    double *m_ptr;
};


class ParamString : public AbstractParam {
    Q_OBJECT
    Q_PROPERTY(QString value READ value WRITE setValue RESET reset NOTIFY valueChanged USER true)
    Q_PROPERTY(bool multiLine READ isMultiLine WRITE setMultiLine)
public:
    ParamString( QObject *parent, const char* name, const QString& value, 
                 bool multiLine, QString *ptr=0 );

    QString value() const { return m_value; }
    void reset() { setValue(m_defaultValue); }
    bool isMultiLine() const { return m_multiLine; }
    void setMultiLine(bool multiLine) { m_multiLine = multiLine; }

signals:
    void valueChanged(const QString&);

public slots:
    void setDirty();
    void setValue(const QString&);

protected:
    QString m_value;
    QString m_defaultValue;
    bool m_multiLine;
    QString *m_ptr;
};


class ParamChoice : public AbstractParam {
    Q_OBJECT
    Q_PROPERTY(QString value READ value WRITE setValue RESET reset NOTIFY valueChanged USER true)
    Q_PROPERTY(QStringList keys READ keys)
    Q_PROPERTY(QVariantList values READ values)
public:
    ParamChoice( QObject *parent, const char* name, const QString& value, const QString& items, QString *ptr=0 );

    QString value() const { return m_value; }
    void reset() { setValue(m_defaultValue); }
    QStringList keys() const { return m_keys; }
    QVariantList values() const { return m_values; }

signals:
    void valueChanged(const QString&);

public slots:
    void setDirty();
    void setValue(const QString&);

protected:
    QString m_value;
    QString m_defaultValue;
    QStringList m_keys;
    QVariantList m_values;
    QString *m_ptr;
};


class ParamEnum : public AbstractParam {
    Q_OBJECT
    Q_PROPERTY(int value READ value WRITE setValue RESET reset NOTIFY valueChanged USER true)
    Q_PROPERTY(QStringList keys READ keys)
    Q_PROPERTY(QVariantList values READ values)
public:                                                      
    ParamEnum( QObject *parent, const char* name, int value, const QString& e, int *ptr=0 );

    int value() const { return m_value; }
    void reset() { setValue(m_defaultValue); }
    QStringList keys() const { return m_keys; }
    QVariantList values() const { return m_values; }

signals:
    void valueChanged(int);

public slots:
    void setDirty();
    void setValue(const QVariant&);
    void setValue(int);

protected:
    int m_value;
    int m_defaultValue;
    QStringList m_keys;
    QVariantList m_values;
    int *m_ptr;
};


class ParamImage : public AbstractParam {
    Q_OBJECT
    Q_PROPERTY(QImage value READ value WRITE setValue RESET reset NOTIFY valueChanged USER true)
public:
    ParamImage( QObject *parent, const char* name, const QImage& value, QImage *ptr=0 );
    ParamImage( QObject *parent, const char* name, const QString& path, QImage *ptr=0 );

    QImage value() const { return m_value; }
    void reset() { setValue(m_defaultValue); }

    virtual void saveValue(QSettings& settings);
    virtual void restoreValue(QSettings& settings);

signals:
    void valueChanged(const QImage&);

public slots:
    void setDirty();
    void setValue(const QImage&);

protected:
    QImage m_value;
    QImage m_defaultValue;
    QImage *m_ptr;
};
