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
#include "fancystyle.h"


static const int windowsItemFrame        =  2; // menu item frame width
static const int windowsSepHeight        =  2; // separator item height
static const int windowsItemHMargin      =  3; // menu item hor text margin
static const int windowsItemVMargin      =  2; // menu item ver text margin
static const int windowsArrowHMargin     =  6; // arrow horizontal margin
static const int windowsTabSpacing       = 12; // space between text and tab
static const int windowsRightBorder      = 15; // right border on windows
static const int windowsCheckMarkWidth   = 12; // checkmarks width on windows

QSize FancyStyle::sizeFromContents( ContentsType type, const QStyleOption *option,
                                    const QSize &size, const QWidget *widget ) const
{
    if (type == CT_MenuItem) {
        if (const QStyleOptionMenuItem *mi = qstyleoption_cast<const QStyleOptionMenuItem *>(option)) {
            QStyleOptionMenuItem opt(*mi);
            opt.maxIconWidth = 20;
            QSize sz = QPlastiqueStyle::sizeFromContents(type, &opt, size, widget);
            sz.rwidth() += 6 - 10 + (windowsItemFrame + windowsItemHMargin) + 5;
            return sz;
        }
    }
    return QPlastiqueStyle::sizeFromContents(type, option, size, widget);
}


void FancyStyle::drawPrimitive( PrimitiveElement element, const QStyleOption *option,
                                QPainter *painter, const QWidget *widget ) const 
{
    QPlastiqueStyle::drawPrimitive(element, option, painter, widget);
}


void FancyStyle::drawControl( ControlElement element, const QStyleOption *option,
                              QPainter *painter, const QWidget *widget) const
{
    QPlastiqueStyle::drawControl(element, option, painter, widget);
}


int FancyStyle::styleHint( StyleHint hint, const QStyleOption *option, const QWidget *widget,
                           QStyleHintReturn *returnData ) const
{
    /*if (hint == SH_ComboBox_Popup) {
        if (const QStyleOptionComboBox *cb = qstyleoption_cast<const QStyleOptionComboBox*>(option)) {
            return !cb->editable;
        }
        return 0;
    }*/
    return QPlastiqueStyle::styleHint(hint, option, widget, returnData);
}


void FancyStyle::polish(QApplication *application) {
    QPlastiqueStyle::polish(application);
    
    #ifndef Q_WS_WIN
    QFont font = QApplication::font();
    font.setPixelSize(11);
    QApplication::setFont(font);
    #endif

    QApplication::setPalette(QApplication::style()->standardPalette());
}


void FancyStyle::polish(QWidget *widget) {
    QPlastiqueStyle::polish(widget);
    {
        QAbstractScrollArea *sa = qobject_cast<QAbstractScrollArea*>(widget);
        if (sa) {
            if (sa->frameShape() == QFrame::StyledPanel) {
                sa->setFrameShadow(QFrame::Plain);
            }
        }
    }
    #ifdef Q_WS_MAC
    {
        QLabel *l = qobject_cast<QLabel*>(widget);
        if (l) {
            l->setFont(QApplication::font());
        }
    }
    #endif
}

	
QPalette FancyStyle::standardPalette() const {
    QPalette palette;

    palette.setBrush(QPalette::Disabled, QPalette::WindowText, QColor(QRgb(0xff808080)));
    palette.setBrush(QPalette::Disabled, QPalette::Button, QColor(QRgb(0xffdddfe4)));
    palette.setBrush(QPalette::Disabled, QPalette::Light, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Disabled, QPalette::Midlight, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Disabled, QPalette::Dark, QColor(QRgb(0xff555555)));
    palette.setBrush(QPalette::Disabled, QPalette::Mid, QColor(QRgb(0xffc7c7c7)));
    palette.setBrush(QPalette::Disabled, QPalette::Text, QColor(QRgb(0xffc7c7c7)));
    palette.setBrush(QPalette::Disabled, QPalette::BrightText, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Disabled, QPalette::ButtonText, QColor(QRgb(0xff808080)));
    palette.setBrush(QPalette::Disabled, QPalette::Base, QColor(QRgb(0xffefefef)));
    palette.setBrush(QPalette::Disabled, QPalette::AlternateBase, palette.color(QPalette::Disabled, QPalette::Base).darker(110));
    palette.setBrush(QPalette::Disabled, QPalette::Window, QColor(QRgb(0xffefefef)));
    palette.setBrush(QPalette::Disabled, QPalette::Shadow, QColor(QRgb(0xff000000)));
    palette.setBrush(QPalette::Disabled, QPalette::Highlight, QColor(QRgb(0xff567594)));
    palette.setBrush(QPalette::Disabled, QPalette::HighlightedText, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Disabled, QPalette::Link, QColor(QRgb(0xff0000ee)));
    palette.setBrush(QPalette::Disabled, QPalette::LinkVisited, QColor(QRgb(0xff52188b)));
    palette.setBrush(QPalette::Active,   QPalette::WindowText, QColor(QRgb(0xff000000)));
    palette.setBrush(QPalette::Active,   QPalette::Button, QColor(QRgb(0xffdddfe4)));
    palette.setBrush(QPalette::Active,   QPalette::Light, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Active,   QPalette::Midlight, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Active,   QPalette::Dark, QColor(QRgb(0xff555555)));
    palette.setBrush(QPalette::Active,   QPalette::Mid, QColor(QRgb(0xffc7c7c7)));
    palette.setBrush(QPalette::Active,   QPalette::Text, QColor(QRgb(0xff000000)));
    palette.setBrush(QPalette::Active,   QPalette::BrightText, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Active,   QPalette::ButtonText, QColor(QRgb(0xff000000)));
    palette.setBrush(QPalette::Active,   QPalette::Base, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Active,   QPalette::AlternateBase, palette.color(QPalette::Active, QPalette::Base).darker(110));
    palette.setBrush(QPalette::Active,   QPalette::Window, QColor(QRgb(0xffefefef)));
    palette.setBrush(QPalette::Active,   QPalette::Shadow, QColor(QRgb(0xff555555)));
    palette.setBrush(QPalette::Active,   QPalette::Highlight, QColor(QRgb(0xff678db2)));
    palette.setBrush(QPalette::Active,   QPalette::HighlightedText, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Active,   QPalette::Link, QColor(QRgb(0xff0000ee)));
    palette.setBrush(QPalette::Active,   QPalette::LinkVisited, QColor(QRgb(0xff52188b)));
    palette.setBrush(QPalette::Inactive, QPalette::WindowText, QColor(QRgb(0xff000000)));
    palette.setBrush(QPalette::Inactive, QPalette::Button, QColor(QRgb(0xffdddfe4)));
    palette.setBrush(QPalette::Inactive, QPalette::Light, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Inactive, QPalette::Midlight, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Inactive, QPalette::Dark, QColor(QRgb(0xff555555)));
    palette.setBrush(QPalette::Inactive, QPalette::Mid, QColor(QRgb(0xffc7c7c7)));
    palette.setBrush(QPalette::Inactive, QPalette::Text, QColor(QRgb(0xff000000)));
    palette.setBrush(QPalette::Inactive, QPalette::BrightText, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Inactive, QPalette::ButtonText, QColor(QRgb(0xff000000)));
    palette.setBrush(QPalette::Inactive, QPalette::Base, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Inactive, QPalette::AlternateBase, palette.color(QPalette::Inactive, QPalette::Base).darker(110));
    palette.setBrush(QPalette::Inactive, QPalette::Window, QColor(QRgb(0xffefefef)));
    palette.setBrush(QPalette::Inactive, QPalette::Shadow, QColor(QRgb(0xff000000)));
    palette.setBrush(QPalette::Inactive, QPalette::Highlight, QColor(QRgb(0xff678db2)));
    palette.setBrush(QPalette::Inactive, QPalette::HighlightedText, QColor(QRgb(0xffffffff)));
    palette.setBrush(QPalette::Inactive, QPalette::Link, QColor(QRgb(0xff0000ee)));
    palette.setBrush(QPalette::Inactive, QPalette::LinkVisited, QColor(QRgb(0xff52188b)));
    return palette;
}

