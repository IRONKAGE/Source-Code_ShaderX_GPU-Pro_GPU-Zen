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
#include "rolloutbox.h"


RolloutBox::RolloutBox(QWidget *parent) : QFrame(parent) {
    m_checkable = false;
    m_checked = true;

    m_layout = new QGridLayout(this);
    m_layout->setContentsMargins(0,0,0,0);
    m_layout->setSpacing(4);
    m_layout->setColumnStretch(0, 50);
    m_layout->setColumnStretch(1, 50);
    m_toolButton = 0;
    m_expanded = true;

    m_toolButton = new QToolButton(this);
    m_layout->addWidget(m_toolButton, 0, 0);
    m_toolButton->setFocusPolicy(Qt::NoFocus);
    m_toolButton->setText("open/close");
    m_toolButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_toolButton->setVisible(false);
    m_toolButton->setAutoRaise(true);
    m_toolButton->setFixedHeight(fontMetrics().height()+6);
    m_toolButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    m_toolButton->setArrowType(Qt::DownArrow);
    QFont font(m_toolButton->font());
    font.setWeight(99);
    m_toolButton->setFont(font);
    connect(m_toolButton, SIGNAL(clicked()), this, SLOT(toggle()));

    m_checkBox = new QCheckBox(this);
    m_checkBox->setChecked(true);
    m_checkBox->setVisible(false);
    m_checkBox->setFixedHeight(20);
    m_checkBox->setText("enable");
    m_layout->addWidget(m_checkBox, 0,1);
    connect(m_checkBox, SIGNAL(toggled(bool)), this, SLOT(setChecked(bool)));
}


void RolloutBox::saveSettings(QSettings& settings) {
    settings.setValue("expanded", isExpanded());
    saveSettings(settings, this);
}


void RolloutBox::restoreSettings(QSettings& settings) {
    setExpanded(settings.value("expanded", true).toBool());
    restoreSettings(settings, this);
}


void RolloutBox::saveSettings(QSettings& settings, QObject *obj) {
    const QObjectList& L = obj->children();
    for (int i = 0; i < L.size(); ++i) {
        RolloutBox *b = qobject_cast<RolloutBox*>(L[i]);
        if (b) {
            settings.beginGroup(b->objectName());
            b->saveSettings(settings);
            settings.endGroup();
        }
    }
}


void RolloutBox::restoreSettings(QSettings& settings, QObject *obj) {
    const QObjectList& L = obj->children();
    for (int i = 0; i < L.size(); ++i) {
        RolloutBox *b = qobject_cast<RolloutBox*>(L[i]);
        if (b) {
            settings.beginGroup(b->objectName());
            b->restoreSettings(settings);
            settings.endGroup();
        }
    }
}


void RolloutBox::setFrame(bool frame) {
    if (frame) {
        setFrameShape(QFrame::StyledPanel);
        m_layout->setContentsMargins(4,4,4,4);
    } else {
        setFrameShape(QFrame::NoFrame);
        m_layout->setContentsMargins(0,0,0,0);
    }
}


void RolloutBox::setTitle(const QString &title) {
    if (!title.isEmpty()) {
        setFrame(true);
    }
    m_toolButton->setText(title);
    m_toolButton->setVisible(!title.isEmpty());
    m_checkBox->setVisible(!title.isEmpty() && m_checkable);
}


void RolloutBox::setCheckable(bool checkable) {
    m_checkable = checkable;
    m_checkBox->setVisible(!m_toolButton->text().isEmpty() && m_checkable);
}


void RolloutBox::setChecked(bool checked) {
    if (m_checked != checked) {
        m_checked = checked;
        m_checkBox->setChecked(checked);
        QList<QWidget*> L = findChildren<QWidget*>();
        for (int i = 0; i < L.size(); ++i) {
            if ((L[i] != m_toolButton) && (L[i] != m_checkBox)) L[i]->setEnabled(checked);
        }
        checkChanged(m_checked);
    }
}


void RolloutBox::setExpanded(bool expanded) {
    if (m_expanded != expanded) {
        QScrollArea *sa = 0;
        QWidget *p = parentWidget();
        while (p) {
            sa = qobject_cast<QScrollArea*>(p);
            if (sa) break;
            p = p->parentWidget();
        }

        bool areUpdatesEnabled;
        if (sa) {
            areUpdatesEnabled = updatesEnabled();
            sa->setUpdatesEnabled(false);
        }
        
        m_expanded = !m_expanded;
        QList<QWidget*> L = findChildren<QWidget*>();
        for (int i = 0; i < L.size(); ++i) {
            if ((L[i] != m_toolButton) && (L[i] != m_checkBox)) L[i]->setVisible(m_expanded);
        }
        parentWidget()->updateGeometry();

        if (sa) {
            qApp->processEvents();
            if (m_expanded) sa->ensureWidgetVisible(this);
            sa->setUpdatesEnabled(true);
        }
        
        if (m_toolButton) {
            m_toolButton->setArrowType(m_expanded? Qt::DownArrow : Qt::RightArrow);
        }

        toggled(m_expanded);
    }
}


void RolloutBox::toggle() {
    setExpanded(!m_expanded);
}


QWidgetList RolloutBox::widgets() const {
    QWidgetList W;
    QList<QWidget*> L = findChildren<QWidget*>();
    for (int i = 0; i < L.size(); ++i) {
        if ((L[i] != m_toolButton) && (L[i] != m_checkBox)) W.append(L[i]);
    }
    return W;
}


void RolloutBox::addWidget(QWidget *w) {
    w->setParent(this);
    m_layout->addWidget(w, m_layout->rowCount(), 0, 1, 2);
}


void RolloutBox::addWidget(QWidget *left, QWidget *right) {
    left->setParent(this);
    right->setParent(this);
    m_layout->addWidget(left, m_layout->rowCount(), 0, Qt::AlignRight);
    m_layout->addWidget(right, m_layout->rowCount()-1, 1);
}
