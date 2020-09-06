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
#include "logwindow.h"
#ifdef _MSC_VER
#include "windows.h"
#endif


static QString g_logBuffer;
static LogWindow *g_logWindow = 0;


class Highlighter : public QSyntaxHighlighter {
public:
    Highlighter(QTextDocument *parent) : QSyntaxHighlighter(parent) {
        m_formatDebug.setForeground(Qt::gray);
        m_formatWarn.setForeground(Qt::red);
        m_formatError.setForeground(Qt::yellow);
        m_formatError.setBackground(Qt::red);
    }

    void highlightBlock(const QString &text) {
        if (text.startsWith("[DEBUG]")) {
            setFormat(0, text.length(), m_formatDebug);
        } 
        else if (text.startsWith("[WARN]")) {  
            setFormat(0, text.length(), m_formatWarn);
        } 
        else if (text.startsWith("[ERROR]")) {
            setFormat(0, text.length(), m_formatError);
        }
        else if (text.startsWith("[FATAL]")) {
            setFormat(0, text.length(), m_formatError);
        }
    }
protected:
    QTextCharFormat m_formatDebug;
    QTextCharFormat m_formatWarn;
    QTextCharFormat m_formatError;
};


LogWindow::LogWindow(QWidget *parent) : QPlainTextEdit(parent) {
    setCenterOnScroll(true);
    setReadOnly(true);
    setLineWrapMode(NoWrap);
    setTextInteractionFlags(Qt::TextSelectableByMouse);

    #ifdef WIN32
    setFont(QFont("Consolas", 8));
    #endif
    new Highlighter(document());
   
    g_logWindow = this;
    appendPlainText(g_logBuffer.trimmed());
    g_logBuffer.clear();    
}


LogWindow::~LogWindow() {
    qInstallMsgHandler(0);
    g_logWindow = 0;
}


void LogWindow::closeEvent(QCloseEvent *e) {
    //QSettings settings;
    //settings.setValue("logWindow/geometry", saveGeometry());
    QPlainTextEdit::closeEvent(e);
} 


void LogWindow::contextMenuEvent(QContextMenuEvent *event) {
    QMenu *menu = createStandardContextMenu();
    QAction *a = new QAction("Clear", this);
    a->setEnabled(!document()->isEmpty());
    connect(a, SIGNAL(triggered()), this, SLOT(clear()));
    QList<QAction*> L = menu->actions();
    menu->insertAction(L[0], a);
    menu->exec(event->globalPos());
    delete menu;
}


void LogWindow::logText(const QString& msg) {
    appendPlainText(msg);
}


static void handle_msg(QtMsgType t, const char * m) {
    QString msg(m);
    QStringList L = msg.split('\n');
    msg = QString();

    QString prefix;
    switch (t) {
        case QtDebugMsg:
            prefix = "[DEBUG] ";
            break;
        case QtWarningMsg:
            prefix = "[WARN]  ";
            break;
        case QtCriticalMsg:
            prefix = "[ERROR] ";
            break;
        case QtFatalMsg:
            prefix = "[FATAL] ";
            break;
    }
    prefix += QDateTime::currentDateTime().toString("hh:mm:ss:zzz");
    //prefix += QString("[%1]").arg((size_t)QThread::currentThreadId(), 8, 16);
    prefix += QString(" - ");

    for (int i = 0; i < L.size(); ++i) {
        msg += prefix + L[i];
        if (i !=  L.size()-1) msg+= "\n";
    }

    #ifdef _MSC_VER
    if (IsDebuggerPresent()) {
        OutputDebugStringA(msg.toStdString().c_str());
        OutputDebugStringA("\n");
    }
    #endif

    if (g_logWindow) {
        if (QThread::currentThread() == qApp->thread()) {
            g_logWindow->logText(msg);
            if (t != QtDebugMsg) {
                if (g_logWindow->isHidden()) g_logWindow->show();
            }
        } else {
            QMetaObject::invokeMethod(g_logWindow, "logText", Qt::QueuedConnection, Q_ARG(QString, msg));
        }
    } else {
        g_logBuffer += msg + "\n";
    }
}


void LogWindow::install() {
    qInstallMsgHandler(handle_msg);
}


