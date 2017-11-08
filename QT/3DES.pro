TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cu \
        ../median.cu

HEADERS += \
    des_constants.h

DISTFILES += \
    Makefile

INCLUDEPATH += /home/swepston/CPE613/cuda8/include
