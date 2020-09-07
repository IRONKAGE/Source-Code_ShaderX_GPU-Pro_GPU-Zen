/******************************************************************************

 @File         PVRShellAPI.h

 @Title        KEGL/PVRShellAPI

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Makes programming for 3D APIs easier by wrapping surface
               initialization, Texture allocation and other functions for use by a demo.

******************************************************************************/

#ifndef __PVRSHELLAPI_H_
#define __PVRSHELLAPI_H_

/****************************************************************************
** 3D API header files
****************************************************************************/
#if defined(__BADA__)
#if defined(BUILD_OGLES)
#include <FGraphicsOpengl.h>
#else
#if defined(BUILD_OGLES2)
#include <FGraphicsOpengl2.h>
#endif
#endif

using namespace Osp::Graphics::Opengl;
#else
#ifdef BUILD_OGLES2
	#include <GLES2/gl2.h>
	#include <EGL/egl.h>
#elif BUILD_OGL
#define SUPPORT_OPENGL
#if defined(WIN32) || defined(UNDER_CE)
	#include <windows.h>
#endif
	#include <GL/gl.h>
	#include <EGL/egl.h>
#elif BUILD_OVG
#include <VG/openvg.h>
#include <EGL/egl.h>
#else
	#include <GLES/egl.h>
#endif
#endif

/*!***************************************************************************
 @Class PVRShellInitAPI
 @Brief Initialisation interface with specific API.
****************************************************************************/
class PVRShellInitAPI
{
public:
	EGLDisplay	m_EGLDisplay;
	EGLSurface	m_EGLWindow;
	EGLContext	m_EGLContext;
	EGLConfig	m_EGLConfig;
	EGLint		m_MajorVersion, m_MinorVersion;
	bool		m_bPowerManagementSupported;
	EGLint		m_iRequestedConfig;

	EGLNativeDisplayType m_NDT;
	EGLNativePixmapType  m_NPT;
	EGLNativeWindowType  m_NWT;

public:
	EGLConfig SelectEGLConfiguration(const PVRShellData * const pData);
	const char *StringFrom_eglGetError() const;
};

#endif // __PVRSHELLAPI_H_

/*****************************************************************************
 End of file (PVRShellAPI.h)
*****************************************************************************/

