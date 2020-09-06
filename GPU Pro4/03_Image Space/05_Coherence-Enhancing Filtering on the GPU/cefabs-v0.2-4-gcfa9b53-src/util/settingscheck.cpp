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
#include "settingscheck.h"
#include "version.h"


void settingsCheck() {
    QSettings settings;
    QString current = settings.value("version").toString();
    if (current != PACKAGE_VERSION) {
        if (current.isEmpty() || 
            QMessageBox::warning(NULL, "Warning", 
                QString("Version of settings is invalid (\"%1\" != \"%2\"). "
                        "Reset settings? (Yes, unless you know what you are doing!)").arg(current).arg(PACKAGE_VERSION),
                QMessageBox::Yes, QMessageBox::No) == QMessageBox::Yes)
        {
            settings.remove("");
        }
        settings.setValue("version", PACKAGE_VERSION);
    }
}
