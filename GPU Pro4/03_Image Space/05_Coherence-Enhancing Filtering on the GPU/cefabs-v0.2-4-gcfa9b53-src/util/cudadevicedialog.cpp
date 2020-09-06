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
#include "cudadevicedialog.h"
#include "ui_cudadevicedialog.h"
#include <cuda_runtime_api.h>


inline int ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    { { 0x10,  8 },
      { 0x11,  8 },
      { 0x12,  8 },
      { 0x13,  8 },
      { 0x20, 32 },
      { 0x21, 48 },
      {   -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    return -1;
}



CudaDeviceDialog::CudaDeviceDialog(QWidget *parent) 
    : QDialog(parent)
{
    m = new Ui_CudaDeviceDialog;
    m->setupUi(this);

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) == cudaSuccess) {
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp p;
            cudaGetDeviceProperties(&p, i);
            m->comboBox->addItem(p.name);
        }
    }
    connect(m->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(updateInfo(int)));
    updateInfo(0);
}


void CudaDeviceDialog::updateInfo(int index) {
    m_infoText = "<html><body>";

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, index);
    if (p.major == 9999 && p.minor == 9999)
        m_infoText += "<p>There is no device supporting CUDA</p>";
    else if (deviceCount == 1)
        m_infoText += "<p>There is 1 device supporting CUDA</p>";
    else
        m_infoText += QString("<p>There are %1 devices supporting CUDA</p>").arg(deviceCount);

    m_infoText += QString("<p>CUDA Driver/Runtime</p>");
    m_infoText += "<table>";
    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    QString error = "<span style='color:red;'>***ERROR*** >= 4.0 required</span>";
    addItem(1, "CUDA Driver Version:", 
               QString("%1.%2 %3").arg(driverVersion/1000).arg(driverVersion%100)
                    .arg((driverVersion >= 4000)? "" : error));
    addItem(1, "CUDA Runtime Version:", 
               QString("%1.%2 %3").arg(runtimeVersion/1000).arg(runtimeVersion%100)
                    .arg((driverVersion >= 4000)? "" : error));
    m_infoText += "</table>";

    if (index < deviceCount) {
        m_infoText += QString("<p>Device %1: &quot;%2&quot;</p>").arg(index).arg(p.name);
        m_infoText += "<table>";
        addItem(1, "CUDA Capability Major/Minor version number:", 
                   QString("%1.%2").arg(p.major).arg(p.minor));

        addItem(1, "Total amount of global memory:", QString("%1 MB").arg(p.totalGlobalMem / 1024 / 1024));

        addItem(1, QString("%1 Multiprocessors x %2 CUDA Cores/MP:").arg(p.multiProcessorCount).arg(ConvertSMVer2Cores(p.major, p.minor)),
                   QString("%1 CUDA Cores").arg(ConvertSMVer2Cores(p.major, p.minor) * p.multiProcessorCount));

        addItem(1, "Total amount of constant memory:", QString("%1 bytes").arg(p.totalConstMem));
        addItem(1, "Total amount of shared memory per block:", QString("%1 bytes").arg(p.sharedMemPerBlock));
        addItem(1, "Total number of registers available per block:", QString("%1").arg(p.regsPerBlock));
        addItem(1, "Warp size:", QString("%1").arg(p.warpSize));
        addItem(1, "Maximum number of threads per block:", QString("%1").arg(p.maxThreadsPerBlock));
        addItem(1, "Maximum sizes of each dimension of a block:", QString("%1 x %2 x %3")
                        .arg(p.maxThreadsDim[0])
                        .arg(p.maxThreadsDim[1])
                        .arg(p.maxThreadsDim[2]));
        addItem(1, "Maximum sizes of each dimension of a grid:", QString("%1 x %2 x %3")
                        .arg(p.maxGridSize[0])
                        .arg(p.maxGridSize[1])
                        .arg(p.maxGridSize[2]));
        addItem(1, "Maximum memory pitch:", QString("%1 bytes").arg(p.memPitch));
        addItem(1, "Texture alignment:", QString("%1 bytes").arg(p.textureAlignment));
        addItem(1, "Clock rate:", QString("%1 GHz").arg(p.clockRate * 1e-6f));
        
        addItem(1, "Concurrent copy and execution:", p.deviceOverlap ? "yes" : "no");
        addItem(1, "# of Asynchronous Copy Engines:", QString("%1").arg(p.asyncEngineCount));
        addItem(1, "Run time limit on kernels:", p.kernelExecTimeoutEnabled ? "yes" : "no");
        addItem(1, "Integrated:", p.integrated ? "yes" : "no");
        addItem(1, "Support host page-locked memory mapping:", p.canMapHostMemory ? "yes" : "no");

        addItem(1, "Compute mode:", p.computeMode == cudaComputeModeDefault ?
                                        "Default (multiple host threads can use this device simultaneously)" :
                                    p.computeMode == cudaComputeModeExclusive ?
                                        "Exclusive (only one host thread at a time can use this device)" :
                                    p.computeMode == cudaComputeModeProhibited ?
                                        "Prohibited (no host thread can use this device)" :
                                        "Unknown");
        addItem(1, "Concurrent kernel execution:", p.concurrentKernels ? "yes" : "no");
        addItem(1, "Device has ECC support enabled:", p.ECCEnabled ? "yes" : "no");
        addItem(1, "Device is using TCC driver mode:", p.tccDriver ? "yes" : "no");

        m_infoText += "</table>";
    }
    m_infoText += "</body></html>";
    m->info->setHtml(m_infoText);

    m->buttonBox->button(QDialogButtonBox::Ok)->setEnabled((driverVersion >= 4000) && (runtimeVersion >= 4000));
}


void CudaDeviceDialog::addItem(int pad, const QString& a, const QString& b) {
    m_infoText += "<tr>";
    m_infoText += QString("<td style='padding-left:%1px;'>").arg(30*pad);
    m_infoText += a;
    m_infoText += "</td><td style='padding-left:10px;'>";
    m_infoText += b;
    m_infoText += "</td>";
    m_infoText += "</tr>";
}


int CudaDeviceDialog::select(bool force) {
    QSettings settings;
    int N = force? -1 : settings.value("cudaDevice", -1).toInt();

    if (N < 0) {
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
            QMessageBox::critical(NULL, "Error", "cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched!");
            exit(1);
        } else {
            if (deviceCount == 0) {
                QMessageBox::critical(NULL, "Error", "No device supporting CUDA found.");
                exit(1);
            }
        }

        CudaDeviceDialog dlg(NULL);
        if (dlg.exec() == QDialog::Accepted) {
            N = dlg.m->comboBox->currentIndex();
        }
    }

    if (N >= 0) {
        settings.setValue("cudaDevice", N);
        cudaSetDevice(N);
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, N);
        qDebug() << "Selected CUDA Device" << N << ":" << p.name;
    }
    return N;
}
