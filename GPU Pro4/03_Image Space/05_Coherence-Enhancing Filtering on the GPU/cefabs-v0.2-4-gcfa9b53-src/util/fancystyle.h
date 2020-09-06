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

class FancyStyle : public QPlastiqueStyle {
public:
    QSize sizeFromContents( ContentsType type, const QStyleOption *option,
                            const QSize &size, const QWidget *widget ) const;

    void drawPrimitive( PrimitiveElement element, const QStyleOption *option,
                        QPainter *painter, const QWidget *widget = 0 ) const;

    void drawControl( ControlElement element, const QStyleOption *option,
                      QPainter *painter, const QWidget *widget) const;

    int styleHint( StyleHint hint, const QStyleOption *option = 0, const QWidget *widget = 0,
                   QStyleHintReturn *returnData = 0 ) const;

    void polish(QApplication *application);
    void polish(QWidget *widget);
    QPalette standardPalette() const;
};
	