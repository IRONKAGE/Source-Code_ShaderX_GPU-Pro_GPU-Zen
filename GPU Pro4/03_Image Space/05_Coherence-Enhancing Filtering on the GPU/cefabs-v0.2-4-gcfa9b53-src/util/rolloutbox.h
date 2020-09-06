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

class RolloutBox : public QFrame {
    Q_OBJECT
public:
    RolloutBox(QWidget *parent);

    void saveSettings(QSettings& settings);
    void restoreSettings(QSettings& settings);

    void setFrame(bool frame);
    QString title() const { return m_toolButton->text(); }
    void setTitle(const QString &title);
    bool isCheckable() const { return m_checkable; }
    void setCheckable(bool checkable);
    bool isChecked() const { return m_checked; }
    bool isExpanded() const { return m_expanded; }

    QWidgetList widgets() const;
    void addWidget(QWidget *w);
    void addWidget(QWidget *left, QWidget *right);

public slots:
    void setChecked(bool checked);
    void setExpanded(bool expanded);
    void toggle();

signals:
    void checkChanged(bool checked = false);
    void toggled(bool);

private:
    QGridLayout *m_layout;
    QToolButton *m_toolButton;
    QCheckBox *m_checkBox;
    bool m_checkable;
    bool m_checked;
    bool m_expanded;

public:
    static void restoreSettings(QSettings& settings, QObject *obj);
    static void saveSettings(QSettings& settings, QObject *obj);
};

