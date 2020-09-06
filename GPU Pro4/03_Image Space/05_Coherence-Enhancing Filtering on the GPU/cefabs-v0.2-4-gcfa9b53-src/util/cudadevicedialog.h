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

class Ui_CudaDeviceDialog;

class CudaDeviceDialog : public QDialog {
    Q_OBJECT
public:
    CudaDeviceDialog(QWidget *parent);
    static int select(bool force=false);

protected slots:
    void updateInfo(int index);

protected:
    void addItem(int pad, const QString& a, const QString& b=QString::null);
    QString m_infoText;
    Ui_CudaDeviceDialog *m;
};
