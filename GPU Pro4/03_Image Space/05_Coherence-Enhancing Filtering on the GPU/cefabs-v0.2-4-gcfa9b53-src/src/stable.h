#ifndef INCLUDED_STABLE_H
#define INCLUDED_STABLE_H

#ifdef __cplusplus

#ifdef WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#endif

#include <cassert>
#include <QtGui/QtGui>
#include <cuda_runtime.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#endif
#endif
