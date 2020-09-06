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
#include "paramui.h"
#include "param.h"
#include <cfloat>


ParamUI::ParamUI(QWidget *parent, QObject *object) : RolloutBox(parent), m_object(object) {
    connect(object, SIGNAL(destroyed(QObject*)), this, SLOT(deleteLater()));
    addObjectParameters(m_object);
}


void ParamUI::addObjectParameters(QObject *obj) {
    const QObjectList& children = obj->children();
    for (int i = 0; i < children.size(); ++i) {
        AbstractParam *p = qobject_cast<AbstractParam*>(children[i]);
        if (!p) continue;
       
        ParamGroup *g = qobject_cast<ParamGroup*>(children[i]);
        if (g) {
            ParamUI *w = new ParamUI(this, g);
            w->setObjectName(g->name());
            addWidget(w);
        } else {
            QVariantHash attr = p->attributes();
            if (attr.contains("keys") && attr.contains("values")) {
                QVariantHash attr = p->attributes();
                QWidget *w = new ParamComboBox(this, p);
                w->setMaximumHeight(20);
                QLabel *l = new ParamLabel(this, p);
                l->setBuddy(w);
                l->setToolTip(p->tooltip());
                addWidget(l,w);
            } else {
                switch (p->type()) {
                    case QVariant::Bool:
                        {
                            QWidget *w = new ParamCheckBox(this, p);
                            QLabel *l = new ParamLabel(this, p);
                            l->setBuddy(w);
                            l->setToolTip(p->tooltip());
                            addWidget(l,w);
                        }
                        break;

                    case QVariant::Int:
                        {
                            QWidget *w = new ParamSpinBox(this, p);
                            QLabel *l = new ParamLabel(this, p);
                            l->setBuddy(w);
                            l->setToolTip(p->tooltip());
                            addWidget(l,w);
                        }
                        break;

                    case QVariant::Double:
                        {
                            QWidget *w = new ParamDoubleSpinBox(this, p);
                            QLabel *l = new ParamLabel(this, p);
                            l->setBuddy(w);
                            l->setToolTip(p->tooltip());
                            addWidget(l,w);
                        }
                        break;

                    case QVariant::String: 
                        {
                            QWidget *w;
                            if (p->property("multiLine").toBool()) 
                                w = new ParamTextEdit(this, p);
                            else
                                w = new ParamLineEdit(this, p);
                            QLabel *l = new QLabel(p->name(), this);
                            l->setBuddy(w);
                            l->setToolTip(p->tooltip());
                            addWidget(l,w);
                        }
                        break;

                    case QVariant::Image: 
                        {
                            QWidget *w = new ParamImageSelect(this, p);
                            QLabel *l = new QLabel(p->name(), this);
                            l->setBuddy(w);
                            l->setToolTip(p->tooltip());
                            addWidget(l,w);
                        }
                        break;

                    default:
                        qWarning() << "Unsupported parameter variant type:" 
                                   << QVariant::typeToName(p->type());
                }
            }
        }
    }

    ParamGroup *g = qobject_cast<ParamGroup*>(obj);
    if (g) {
        setTitle(g->name());
        if (g->type() == QVariant::Bool) {
            setCheckable(true);
            setChecked(g->value());
            connect(this, SIGNAL(checkChanged(bool)), g, SLOT(setValue(bool)));
            connect(g, SIGNAL(valueChanged(bool)), this, SLOT(setChecked(bool)));
        }
    }
}


ParamLabel::ParamLabel(QWidget *parent, AbstractParam *param) : QLabel(param->name(), parent) {
    setFixedHeight(fontMetrics().height()+8);
    m_param = param;
    connect(param, SIGNAL(destroyed(QObject*)), this, SLOT(deleteLater()));
}

    
void ParamLabel::mouseDoubleClickEvent(QMouseEvent * e) {
    if (m_param) {
        m_param->resetValue();
    }
}


ParamCheckBox::ParamCheckBox(QWidget *parent, AbstractParam *param) : QCheckBox(parent) {
    m_param = param;
    setFixedHeight(fontMetrics().height()+8);
    setChecked(param->value().toBool());
    connect(param, SIGNAL(destroyed(QObject*)), this, SLOT(deleteLater()));
    connect(param, SIGNAL(valueChanged(bool)), this, SLOT(setChecked(bool)));
    connect(this, SIGNAL(toggled(bool)), param, SLOT(setValue(bool)));
}


bool ParamCheckBox::event(QEvent *e) {
    if (!isEnabled()) {
        switch(e->type()) {
            case QEvent::MouseButtonPress:
            case QEvent::MouseButtonRelease:
            case QEvent::MouseMove:
                e->ignore();
                return false;
            default:
                break;
        }
    }
    return QCheckBox::event(e);
}


ParamSpinBox::ParamSpinBox(QWidget *parent, AbstractParam *param) : QSpinBox(parent) {
    m_param = param;
    setFixedHeight(fontMetrics().height()+8);
    QVariantHash attr = param->attributes();
    int rmin = attr.value("minimum", -INT_MAX).toInt();
    int rmax = attr.value("maximum", INT_MAX).toInt();
    setKeyboardTracking(false);
    setRange(rmin, rmax);
    setSingleStep(attr.value("singleStep", 1).toInt());
    //setButtonSymbols(QSpinBox::NoButtons);
    setValue(param->value().toInt());
    connect(param, SIGNAL(destroyed(QObject*)), this, SLOT(deleteLater()));
    connect(param, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));
    connect(this, SIGNAL(valueChanged(int)), param, SLOT(setValue(int)));
}


ParamDoubleSpinBox::ParamDoubleSpinBox(QWidget *parent, AbstractParam *param) : QDoubleSpinBox(parent) {
    m_param = param;
    setFixedHeight(fontMetrics().height()+8);
    QVariantHash attr = param->attributes();
    double rmin = attr.value("minimum", -FLT_MAX).toDouble();
    double rmax = attr.value("maximum", FLT_MAX).toDouble();
    setKeyboardTracking(false);
    setRange(rmin, rmax);
    setSingleStep(attr.value("singleStep", 1).toDouble());
    setDecimals(attr.value("decimals", 5).toInt());
    //setButtonSymbols(QDoubleSpinBox::NoButtons);
    setValue(param->value().toDouble());
    connect(param, SIGNAL(destroyed(QObject*)), this, SLOT(deleteLater()));
    connect(param, SIGNAL(valueChanged(double)), this, SLOT(setValue(double)));
    connect(this, SIGNAL(valueChanged(double)), param, SLOT(setValue(double)));
}


ParamComboBox::ParamComboBox(QWidget *parent, AbstractParam *param) : QComboBox(parent) {
    m_param = param;
    setFixedHeight(fontMetrics().height()+8);
    QVariantHash attr = param->attributes();
    QStringList keys = attr["keys"].toStringList();
    QVariantList values = attr["values"].toList();
    for (int i = 0; i < keys.size(); ++i) {
        addItem(keys[i], values[i]);
    }
    setValue(param->value());
    connect(param, SIGNAL(destroyed(QObject*)), this, SLOT(deleteLater()));
    connect(param, SIGNAL(valueChanged(const QVariant&)), this, SLOT(setValue(const QVariant&)));
    connect(this, SIGNAL(activated(int)), this, SLOT(updateParam(int)));
}



void ParamComboBox::setValue(const QVariant& value) {
    int current = findData(value, Qt::UserRole);
    if (current != -1) {
        setCurrentIndex(current);
    }
}


void ParamComboBox::updateParam(int index) {
    QVariant value = itemData(index, Qt::UserRole);
    m_param->setValue(value);
}


ParamLineEdit::ParamLineEdit(QWidget *parent, AbstractParam *param) : QLineEdit(parent) {
    m_param = param;
    setFixedHeight(fontMetrics().height()+8);
    setText(param->value().toString());
    connect(param, SIGNAL(destroyed(QObject*)), this, SLOT(deleteLater()));
    connect(param, SIGNAL(valueChanged(const QString&)), this, SLOT(setText(const QString&)));
    connect(this, SIGNAL(editingFinished()), this, SLOT(updateParam()));
}

void ParamLineEdit::updateParam() {
    m_param->setValue(text());
}


ParamTextEdit::ParamTextEdit(QWidget *parent, AbstractParam *param) : QToolButton(parent) {
    m_param = param;
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    setFixedHeight(fontMetrics().height()+8);
    setText("...");
    setToolTip(m_param->value().toString());
    connect(param, SIGNAL(destroyed(QObject*)), this, SLOT(deleteLater()));
    connect(this, SIGNAL(clicked()), this, SLOT(edit()));
}


void ParamTextEdit::edit() {
    QDialog dlg(this);
    dlg.setWindowTitle("Edit Text");
    dlg.setMinimumHeight(300);
    dlg.setMinimumWidth(500);
    QVBoxLayout *vbox = new QVBoxLayout(&dlg);
    QPlainTextEdit *textEdit = new QPlainTextEdit(&dlg);
    textEdit->setPlainText(m_param->value().toString());
    vbox->addWidget(textEdit);
    QDialogButtonBox *bbox = new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel,Qt::Horizontal, &dlg);
    vbox->addWidget(bbox);
    connect(bbox, SIGNAL(accepted()), &dlg, SLOT(accept()));
    connect(bbox, SIGNAL(rejected()), &dlg, SLOT(reject()));
    if (dlg.exec() == QDialog::Accepted) {
        QString text = textEdit->toPlainText();
        m_param->setValue(text);
        setToolTip(text);
    }
}


namespace{
    class ImageLabel : public QLabel {
    public:            
        ImageLabel(QWidget *parent) {
            QSizePolicy szp(QSizePolicy::Ignored, QSizePolicy::Ignored);
            szp.setHeightForWidth(true);
            setSizePolicy(szp);
        }

        int heightForWidth(int w) const {
            const QPixmap *pm = pixmap();
            if (!pm) return 48;
            QSize sz = pm->size();
            if (sz.width() == 0) return 48;
            int h = w * sz.height() / sz.width();
            return h;
        }
    };
}


ParamImageSelect::ParamImageSelect(QWidget *parent, AbstractParam *param) : QWidget(parent) {
    m_param = param;

    m_image = m_param->value().value<QImage>();
    m_filename = m_image.text("filename");
    
    QHBoxLayout *hbox = new QHBoxLayout(this);
    hbox->setContentsMargins(0,0,0,0);
    hbox->setSpacing(4);

    m_label = new ImageLabel(this);
    m_label->setFrameStyle(QFrame::StyledPanel);
    m_label->setScaledContents(true);
    m_label->setPixmap(QPixmap::fromImage(m_image));
    m_label->setToolTip(m_filename);
    hbox->addWidget(m_label);
    hbox->setStretchFactor(m_label, 100);

    QVBoxLayout *vbox = new QVBoxLayout();
    hbox->addLayout(vbox);
    vbox->setSpacing(4);

    m_edit = new QToolButton(this);
    QPalette p = m_edit->palette();
    p.setBrush(QPalette::Background, p.button());
    m_edit->setPalette(p);
    m_edit->setFixedSize(16,16);
    m_edit->setIcon(style()->standardIcon(QStyle::SP_TitleBarUnshadeButton));
    vbox->addWidget(m_edit);
    m_edit->setText("...");

    m_clear = new QToolButton(this);
    m_clear->setPalette(p);
    m_clear->setFixedSize(16,16);
    m_clear->setIcon(style()->standardIcon(QStyle::SP_TitleBarCloseButton));
    vbox->addWidget(m_clear);
    vbox->addStretch();

    connect(param, SIGNAL(destroyed(QObject*)), this, SLOT(deleteLater()));
    connect(param, SIGNAL(valueChanged(const QImage&)), this, SLOT(setImage(const QImage&)));
    connect(m_edit, SIGNAL(clicked()), this, SLOT(edit()));
    connect(m_clear, SIGNAL(clicked()), this, SLOT(clear()));
}


void ParamImageSelect::setImage(const QImage& image) {
    if (m_image != image) {
        m_filename = image.text("filename");
        m_image = image;
        m_label->setToolTip(m_filename);
        m_label->setPixmap(QPixmap::fromImage(image));
        m_param->setValue(image);
    }
}


void ParamImageSelect::clear() {
    m_image = QImage();
    m_label->setToolTip("");
    m_label->setPixmap(QPixmap());
    m_param->setValue(m_image);
}


void ParamImageSelect::edit() {
    QString filename = QFileDialog::getOpenFileName(this, "Open", m_filename,
        "Images (*.png *.bmp *.jpg *.jpeg);;All files (*.*)");
    if (!filename.isEmpty()) {
        QImage image(filename);
        if (image.isNull()) {
            QMessageBox::critical(this, "Error", QString("Loading '%1' failed!").arg(filename));
            return;
        }
        image.setText("filename", filename);
        setImage(image);

        QScrollArea *sa = 0;
        QWidget *p = parentWidget();
        while (p) {
            sa = qobject_cast<QScrollArea*>(p);
            if (sa) break;
            p = p->parentWidget();
        }
        if (sa) sa->ensureWidgetVisible(m_label, 5, 5);
    }
}
