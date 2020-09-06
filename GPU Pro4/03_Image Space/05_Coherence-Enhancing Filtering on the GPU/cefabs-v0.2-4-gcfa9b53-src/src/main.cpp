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
#include "mainwindow.h"
#include "fancystyle.h"
#include "settingscheck.h"
#include "cudadevicedialog.h"
#include "logwindow.h"


int main(int argc, char **argv) {
    LogWindow::install();
    QApplication::setOrganizationDomain("jkyprian.hpi3d.de");
    QApplication::setOrganizationName("jkyprian");
    QApplication::setApplicationName("cefabs");
    QApplication app(argc, argv);
    QApplication::setStyle(new FancyStyle);
    settingsCheck();
    if (CudaDeviceDialog::select() < 0) return 1;

    MainWindow mw;
    mw.showNormal();
    app.connect(&app, SIGNAL(lastWindowClosed()), &app, SLOT(quit()));
    int result = app.exec();
    return 0;
}
