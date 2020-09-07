/******************************************************************************

 @File         PVRShellAPI.cpp

 @Title        KEGL/PVRShellAPI

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Makes programming for 3D APIs easier by wrapping surface
               initialization, Texture allocation and other functions for use by a demo.

******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "PVRShell.h"
#include "PVRShellAPI.h"
#include "PVRShellOS.h"
#include "PVRShellImpl.h"

#ifndef EGL_CONTEXT_LOST_IMG
/*! Extended error code EGL_CONTEXT_LOST_IMG generated when power management event has occurred. */
#define EGL_CONTEXT_LOST_IMG		0x300E
#endif

#ifndef EGL_CONTEXT_PRIORITY_LEVEL_IMG
/*! An extensions added to the list of attributes for the context to give it a priority hint */
#define EGL_CONTEXT_PRIORITY_LEVEL_IMG		0x3100
/*! Request the context is created with high priority */
#define EGL_CONTEXT_PRIORITY_HIGH_IMG		0x3101
/*! Request the context is created with medium priority */
#define EGL_CONTEXT_PRIORITY_MEDIUM_IMG		0x3102
/*! Request the context is created with low priority */
#define EGL_CONTEXT_PRIORITY_LOW_IMG		0x3103
#endif

/*****************************************************************************
	Declarations
*****************************************************************************/
static bool PVRShellIsExtensionSupported(EGLDisplay dpy, const char *extension);

#if defined BUILD_OVG && !defined EGL_VERSION_1_2
#error OpenVG requires egl.h version 1.2 or higher
#endif

#if defined GL_ES_VERSION_2_0 && !defined EGL_VERSION_1_3 && !defined BUILD_OVG
#error OpenGL ES 2 requires egl.h version 1.3 or higher
#endif

/****************************************************************************
** Class: PVRShellInitAPI
****************************************************************************/

/*****************************************************************************
* Function Name  : ActivatePreferences
* Description    : Activates the user set preferences (like v-sync)
*****************************************************************************/
void PVRShellInit::ApiActivatePreferences()
{

#ifdef EGL_VERSION_1_1
	eglSwapInterval(m_EGLDisplay, m_pShell->m_pShellData->nSwapInterval);
#endif

#ifdef BUILD_OVG
	vgSeti(VG_RENDERING_QUALITY, VG_RENDERING_QUALITY_NONANTIALIASED + m_pShell->m_pShellData->nFSAAMode);
#endif
}

/*****************************************************************************
* Function Name  : ApiInitAPI
* Returns        : true for success
* Description    : Initialise the 3D API
*****************************************************************************/
bool PVRShellInit::ApiInitAPI()
{
	int					bDone;

	m_NDT = (EGLNativeDisplayType)OsGetNativeDisplayType();
	m_NPT = (EGLNativePixmapType) OsGetNativePixmapType();
	m_NWT = (EGLNativeWindowType) OsGetNativeWindowType();

	m_EGLContext = 0;

	do
	{
		bDone = true;

		m_EGLDisplay = eglGetDisplay(m_NDT);

		if(m_EGLDisplay == EGL_NO_DISPLAY)
		{
#if defined(BUILD_OGLES2) || defined(BUILD_OVG)
			m_EGLDisplay = eglGetDisplay((EGLNativeDisplayType)EGL_DEFAULT_DISPLAY);
#else
			m_EGLDisplay = eglGetDisplay((NativeDisplayType)EGL_DEFAULT_DISPLAY);
#endif
		}

		if(!eglInitialize(m_EGLDisplay, &m_MajorVersion, &m_MinorVersion))
		{
			m_pShell->PVRShellSet(prefExitMessage, "PVRShell: Unable to initialise EGL\n");
			m_pShell->PVRShellOutputDebug("PVRShell: EGL Error (%s)\n", StringFrom_eglGetError());
			return false;
		}

		m_pShell->PVRShellOutputDebug("PVRShell: EGL %d.%d initialized\n", m_MajorVersion, m_MinorVersion);

		// Check Extension avaliablility after EGL initialization
		if (m_MajorVersion > 1 || (m_MajorVersion == 1 && m_MinorVersion >= 1))
		{
			m_bPowerManagementSupported = true;
		}
		else
		{
			m_bPowerManagementSupported = PVRShellIsExtensionSupported(m_EGLDisplay,"EGL_IMG_power_management");
		}

		do
		{
			// bind OpenVG API if requested
			if(m_pShell->m_pShellData->bNeedOpenVG)
			{
#ifndef BUILD_OVG
				m_pShell->PVRShellSet(prefExitMessage, "PVRShell: OpenVG not supported. Compile with BUILD_OVG defined.");
				return false;
#else
				if(!eglBindAPI(EGL_OPENVG_API))
				{
					m_pShell->PVRShellSet(prefExitMessage, "PVRShell: Failed to bind OpenVG API\n");
					return false;
				}

				m_pShell->PVRShellOutputDebug("PVRShell: OpenVG bound\n");
#endif
			}
			else
			{
#if defined(BUILD_OGL)
				if(!eglBindAPI(EGL_OPENGL_API))
				{
					m_pShell->PVRShellSet(prefExitMessage, "PVRShell: Failed to bind OpenGL API\n");
					return false;
				}
#else
#if defined EGL_VERSION_1_3 && defined GL_ES_VERSION_2_0

				if(!eglBindAPI(EGL_OPENGL_ES_API))
				{
					m_pShell->PVRShellSet(prefExitMessage, "PVRShell: Failed to bind OpenGL ES API\n");
					return false;
				}
#endif
#endif
			}

			// Find an EGL config
			m_EGLConfig = SelectEGLConfiguration(m_pShell->m_pShellData);

			// Destroy the context if we already created one
			if (m_EGLContext)
			{
				eglDestroyContext(m_EGLDisplay, m_EGLContext);
			}

			// Attempt to create a context
			EGLint ai32ContextAttribs[32];
			int	i = 0;

#if defined(EGL_VERSION_1_3) && defined(GL_ES_VERSION_2_0)
			ai32ContextAttribs[i++] = EGL_CONTEXT_CLIENT_VERSION;
			ai32ContextAttribs[i++] = 2;
#endif

#if defined(BUILD_OGLES2) || defined(BUILD_OGLES)
			if(PVRShellIsExtensionSupported(m_EGLDisplay,"EGL_IMG_context_priority"))
			{
				ai32ContextAttribs[i++] = EGL_CONTEXT_PRIORITY_LEVEL_IMG;
				switch(m_pShell->PVRShellGet(prefPriority))
				{
					case 0: ai32ContextAttribs[i++] = EGL_CONTEXT_PRIORITY_LOW_IMG; break;
					case 1: ai32ContextAttribs[i++] = EGL_CONTEXT_PRIORITY_MEDIUM_IMG; break;
					default:ai32ContextAttribs[i++] = EGL_CONTEXT_PRIORITY_HIGH_IMG;
				}
			}
#endif
			ai32ContextAttribs[i] = EGL_NONE;

			m_EGLContext = eglCreateContext(m_EGLDisplay, m_EGLConfig, NULL, ai32ContextAttribs);

			if(m_EGLContext == EGL_NO_CONTEXT)
			{
				if(m_iRequestedConfig > 0)
				{
					// We failed to create a context
					m_pShell->PVRShellSet(prefExitMessage, "PVRShell: Unable to create a context\n");
					return false;
				}
				else if(m_pShell->m_pShellData->bNeedPbuffer)
				{
					// Disable P-buffer and try again
					m_pShell->m_pShellData->bNeedPbuffer = false;
				}
				else if(m_pShell->m_pShellData->bNeedStencilBuffer)
				{
                    // Disable Stencil Buffer and try again
					m_pShell->m_pShellData->bNeedStencilBuffer = false;
				}
				else if(m_pShell->m_pShellData->nFSAAMode > 0)
				{
					// Still failing, reduce the mode of AA and try again
					--m_pShell->m_pShellData->nFSAAMode;
				}
				else
				{
					m_pShell->PVRShellSet(prefExitMessage, "PVRShell: Unable to create a context\n");
					return false;
				}
			}
		} while(m_EGLContext == EGL_NO_CONTEXT);

		EGLint		attrib_list[16];
		int	i = 0;
#if defined(EGL_VERSION_1_2)
		if(m_pShell->m_pShellData->bNeedAlphaFormatPre) // The default is EGL_ALPHA_FORMAT_NONPRE
		{
			attrib_list[i++] = EGL_ALPHA_FORMAT;
			attrib_list[i++] = EGL_ALPHA_FORMAT_PRE;
		}
#endif
		// Terminate the attribute list with EGL_NONE
		attrib_list[i] = EGL_NONE;

		if(m_pShell->m_pShellData->bNeedPixmap)
		{
			m_pShell->PVRShellOutputDebug("InitAPI() Using pixmaps, about to create egl surface\n");
			m_EGLWindow = eglCreatePixmapSurface(m_EGLDisplay, m_EGLConfig, m_NPT, attrib_list);
		}
		else
		{
			m_EGLWindow = eglCreateWindowSurface(m_EGLDisplay, m_EGLConfig, m_NWT, attrib_list);

            // If we have failed to create a surface then try using Null
			if(m_EGLWindow == EGL_NO_SURFACE)
			{
				m_EGLWindow = eglCreateWindowSurface(m_EGLDisplay, m_EGLConfig, NULL, attrib_list);
			}
		}

		if (m_EGLWindow == EGL_NO_SURFACE)
		{
			m_pShell->PVRShellSet(prefExitMessage, "PVRShell: Unable to create surface\n");
			return false;
		}

		if (!eglMakeCurrent(m_EGLDisplay, m_EGLWindow, m_EGLWindow, m_EGLContext))
		{
#ifdef EGL_VERSION_1_3
			if((eglGetError() == EGL_CONTEXT_LOST))
#else
			if((eglGetError() == EGL_CONTEXT_LOST_IMG) && m_bPowerManagementSupported)
#endif
			{
				bDone = false;
			}
			else
			{
				m_pShell->PVRShellSet(prefExitMessage, "PVRShell: Unable to make context current\n");
				return false;
			}
		}
	} while(!bDone);

	/*
	 	Get correct screen width and height and
		save them into
		m_pShell->m_pShellData->nShellDimX and
		m_pShell->m_pShellData->nShellDimY
	*/
	eglQuerySurface(m_EGLDisplay, m_EGLWindow,
			EGL_WIDTH,  (EGLint*)&m_pShell->m_pShellData->nShellDimX
		);
	eglQuerySurface(m_EGLDisplay, m_EGLWindow,
			EGL_HEIGHT, (EGLint*)&m_pShell->m_pShellData->nShellDimY
		);

	/*
		Done - activate requested features
	*/
	ApiActivatePreferences();
	return true;
}

/*!***********************************************************************
@Function		OutputAPIInfo
@description	When prefOutputInfo is set to true this function outputs
				various pieces of API dependent information via
				PVRShellOutputDebug.
*************************************************************************/
void PVRShellInit::OutputAPIInfo()
{
	// Output API dependent information
	if(m_pShell->PVRShellGet(prefOutputInfo))
	{
		EGLint i32Values[5];

		m_pShell->PVRShellOutputDebug("\n");
#ifndef BUILD_OVG
		m_pShell->PVRShellOutputDebug("GL:\n");
		m_pShell->PVRShellOutputDebug("  Vendor:   %s\n", (char*) glGetString(GL_VENDOR));
		m_pShell->PVRShellOutputDebug("  Renderer: %s\n", (char*) glGetString(GL_RENDERER));
		m_pShell->PVRShellOutputDebug("  Version:  %s\n", (char*) glGetString(GL_VERSION));
#else
		m_pShell->PVRShellOutputDebug("VG:\n");
		m_pShell->PVRShellOutputDebug("  Vendor:   %s\n", (char*) vgGetString(VG_VENDOR));
		m_pShell->PVRShellOutputDebug("  Renderer: %s\n", (char*) vgGetString(VG_RENDERER));
		m_pShell->PVRShellOutputDebug("  Version:  %s\n", (char*) vgGetString(VG_VERSION));
#endif

		m_pShell->PVRShellOutputDebug("\n");
		m_pShell->PVRShellOutputDebug("EGL:\n");
		m_pShell->PVRShellOutputDebug("  Vendor:   %s\n" , (char*) eglQueryString(m_EGLDisplay, EGL_VENDOR));
		m_pShell->PVRShellOutputDebug("  Version:  %s\n" , (char*) eglQueryString(m_EGLDisplay, EGL_VERSION));

		if(eglQueryContext(m_EGLDisplay, m_EGLContext, EGL_CONTEXT_PRIORITY_LEVEL_IMG, &i32Values[0]))
		{
			switch(i32Values[0])
			{
				case EGL_CONTEXT_PRIORITY_HIGH_IMG:   m_pShell->PVRShellOutputDebug("  Context priority: High\n");  break;
				case EGL_CONTEXT_PRIORITY_MEDIUM_IMG: m_pShell->PVRShellOutputDebug("  Context priority: Medium\n");break;
				case EGL_CONTEXT_PRIORITY_LOW_IMG:    m_pShell->PVRShellOutputDebug("  Context priority: Low\n");   break;
				default: m_pShell->PVRShellOutputDebug("  Context priority: Unrecognised.\n");
			}
		}
		else
		{
			eglGetError(); // Clear error
			m_pShell->PVRShellOutputDebug("  Context priority: Unsupported\n");
		}

#ifdef EGL_VERSION_1_2
		m_pShell->PVRShellOutputDebug("  Client APIs:  %s\n" , (char*) eglQueryString(m_EGLDisplay, EGL_CLIENT_APIS));
#endif

		m_pShell->PVRShellOutputDebug("\n");
		m_pShell->PVRShellOutputDebug("Window Width:  %i\n" , m_pShell->PVRShellGet(prefWidth));
		m_pShell->PVRShellOutputDebug("Window Height: %i\n" , m_pShell->PVRShellGet(prefHeight));
		m_pShell->PVRShellOutputDebug("Is Rotated: %s\n", m_pShell->PVRShellGet(prefIsRotated) ? "Yes" : "No");
		m_pShell->PVRShellOutputDebug("\n");

		// Colour buffer
		m_pShell->PVRShellOutputDebug("EGL Surface:\n");
		eglGetConfigAttrib(m_EGLDisplay, m_EGLConfig, EGL_BUFFER_SIZE , &i32Values[0]);
		eglGetConfigAttrib(m_EGLDisplay, m_EGLConfig, EGL_RED_SIZE    , &i32Values[1]);
		eglGetConfigAttrib(m_EGLDisplay, m_EGLConfig, EGL_GREEN_SIZE  , &i32Values[2]);
		eglGetConfigAttrib(m_EGLDisplay, m_EGLConfig, EGL_BLUE_SIZE   , &i32Values[3]);
		eglGetConfigAttrib(m_EGLDisplay, m_EGLConfig, EGL_ALPHA_SIZE  , &i32Values[4]);

		m_pShell->PVRShellOutputDebug("  Colour Buffer:  %i bits (R%i G%i B%i A%i)\n", i32Values[0],i32Values[1],i32Values[2],i32Values[3],i32Values[4]);

		// Depth buffer
		eglGetConfigAttrib(m_EGLDisplay, m_EGLConfig, EGL_DEPTH_SIZE , &i32Values[0]);
		m_pShell->PVRShellOutputDebug("  Depth Buffer:   %i bits\n", i32Values[0]);

		// Stencil Buffer
		eglGetConfigAttrib(m_EGLDisplay, m_EGLConfig, EGL_STENCIL_SIZE , &i32Values[0]);
		m_pShell->PVRShellOutputDebug("  Stencil Buffer: %i bits\n", i32Values[0]);

		// EGL surface bits support
		eglGetConfigAttrib(m_EGLDisplay, m_EGLConfig, EGL_SURFACE_TYPE , &i32Values[0]);
		m_pShell->PVRShellOutputDebug("  Surface type:   %s%s%s\n",	i32Values[0] & EGL_WINDOW_BIT  ? "WINDOW " : "",
																		i32Values[1] & EGL_PBUFFER_BIT ? "PBUFFER " : "",
																		i32Values[2] & EGL_PIXMAP_BIT  ? "PIXMAP " : "");
		// EGL renderable type
#ifdef EGL_VERSION_1_2
		eglGetConfigAttrib(m_EGLDisplay, m_EGLConfig, EGL_RENDERABLE_TYPE , &i32Values[0]);
		m_pShell->PVRShellOutputDebug("  Renderable type: %s%s%s%s\n", i32Values[0] & EGL_OPENVG_BIT ? "OPENVG " : "",
															i32Values[0] & EGL_OPENGL_ES_BIT ? "OPENGL_ES " : "",
#ifdef EGL_OPENGL_BIT
															i32Values[0] & EGL_OPENGL_BIT ? "OPENGL " :
#endif
															"",
															i32Values[0] & EGL_OPENGL_ES2_BIT ? "OPENGL_ES2 " : "");
#endif

#ifndef BUILD_OVG
		eglGetConfigAttrib(m_EGLDisplay, m_EGLConfig, EGL_SAMPLE_BUFFERS , &i32Values[0]);
		eglGetConfigAttrib(m_EGLDisplay, m_EGLConfig, EGL_SAMPLES , &i32Values[1]);
		m_pShell->PVRShellOutputDebug("  Sample buffer No.: %i\n", i32Values[0]);
		m_pShell->PVRShellOutputDebug("  Samples per pixel: %i\n", i32Values[1]);
#else
		m_pShell->PVRShellOutputDebug("\n");

		switch(vgGeti(VG_RENDERING_QUALITY))
		{
			case VG_RENDERING_QUALITY_BETTER:
				m_pShell->PVRShellOutputDebug("Rendering quality:  VG_RENDERING_QUALITY_BETTER\n");
			break;
			case VG_RENDERING_QUALITY_FASTER:
				m_pShell->PVRShellOutputDebug("Rendering quality:  VG_RENDERING_QUALITY_FASTER\n");
			break;
			default:
				m_pShell->PVRShellOutputDebug("Rendering quality:  VG_RENDERING_QUALITY_NONANTIALIASED\n");
		}

#endif
	}
}

/*!***********************************************************************
 @Function		ApiReleaseAPI
 @description	Releases all resources allocated by the API.
*************************************************************************/
void PVRShellInit::ApiReleaseAPI()
{
	eglSwapBuffers(m_EGLDisplay, m_EGLWindow);
	eglMakeCurrent(m_EGLDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
 	eglDestroyContext(m_EGLDisplay, m_EGLContext);
	eglDestroySurface(m_EGLDisplay, m_EGLWindow);
   	eglTerminate(m_EGLDisplay);
}

/*******************************************************************************
 * Function Name  : SelectEGLConfiguration
 * Inputs		  : pData
 * Returns        : EGLConfig
 * Description    : Find the config to use for EGL initialisation
 *******************************************************************************/
EGLConfig PVRShellInitAPI::SelectEGLConfiguration(const PVRShellData * const pData)
{
    EGLint		num_config;
    EGLint		conflist[32];
	EGLConfig	conf = (EGLConfig) 0;
    int			i = 0;

	// Specific config ID requested?
	if (m_iRequestedConfig > 0)
	{
		conflist[i++] = EGL_CONFIG_ID;
		conflist[i++] = m_iRequestedConfig;
		conflist[i++] = EGL_NONE;
		if(!eglChooseConfig(m_EGLDisplay, conflist, &conf, 1, &num_config) || num_config != 1)
		{
			return 0;
		}
		return conf;
	}

	// Select default configuration
    conflist[i++] = EGL_LEVEL;
    conflist[i++] = 0;

#if !defined(ANDROID_MATCH_PIXEL_FORMAT)
	conflist[i++] = EGL_BUFFER_SIZE;
	conflist[i++] = pData->nColorBPP;
#endif

	if(pData->bNeedZbuffer)
	{
#ifndef BUILD_OVG
		conflist[i++] = EGL_DEPTH_SIZE;
		conflist[i++] = 16;
#endif
	}

	if(pData->bNeedStencilBuffer)
	{
		conflist[i++] = EGL_STENCIL_SIZE;
		conflist[i++] = 8;
	}

	conflist[i++] = EGL_SURFACE_TYPE;
	conflist[i] = EGL_WINDOW_BIT;

	if(pData->bNeedPbuffer)
	{
		conflist[i] |= EGL_PBUFFER_BIT;
	}

	i++;

#ifdef BUILD_OVG
	conflist[i++] = EGL_RENDERABLE_TYPE;
	conflist[i++] = EGL_OPENVG_BIT;
#elif BUILD_OGL
	conflist[i++] = EGL_RENDERABLE_TYPE;
	conflist[i++] = EGL_OPENGL_BIT;
#elif defined EGL_VERSION_1_3 && GL_ES_VERSION_2_0
	conflist[i++] = EGL_RENDERABLE_TYPE;
	conflist[i++] = EGL_OPENGL_ES2_BIT;
#endif

#ifndef BUILD_OVG
	// Append number of number buffers depending on FSAA mode selected
	switch(pData->nFSAAMode)
	{
		case 1:
			conflist[i++] = EGL_SAMPLE_BUFFERS;
			conflist[i++] = 1;
			conflist[i++] = EGL_SAMPLES;
			conflist[i++] = 2;
		break;

		case 2:
			conflist[i++] = EGL_SAMPLE_BUFFERS;
			conflist[i++] = 1;
			conflist[i++] = EGL_SAMPLES;
			conflist[i++] = 4;
		break;

		default:
			conflist[i++] = EGL_SAMPLE_BUFFERS;
			conflist[i++] = 0;
	}
#endif

#if defined(__BADA__)
	conflist[i++] = EGL_RED_SIZE;
	conflist[i++] = 5;
	conflist[i++] = EGL_GREEN_SIZE;
	conflist[i++] = 6;
	conflist[i++] = EGL_BLUE_SIZE;
	conflist[i++] = 5;
	conflist[i++] = EGL_ALPHA_SIZE;
	conflist[i++] = 0;
#endif

#if defined(ANDROID_MATCH_PIXEL_FORMAT)
	if(m_NWT != NULL)
	{
		EGLint r,g,b,a, value;
		int i32Total_num_configs, j;
		EGLConfig	*pConfigs;
		int format;

		if(m_NWT->query(m_NWT, NATIVE_WINDOW_FORMAT, &format))
		{
			return 0; // Failed to query for format
		}

		// On android the egl config has to have the same pixel format as the native window because
		// pixel format conversion is prohibited.
		switch(format)
		{
			case HAL_PIXEL_FORMAT_RGBA_8888:
			case HAL_PIXEL_FORMAT_BGRA_8888:
				r = g = b = a = 8;
			break;
			case HAL_PIXEL_FORMAT_RGB_888:
				r = g = b = 8; a = 0;
			break;
			case HAL_PIXEL_FORMAT_RGB_565:
				r = 5; g = 6; b = 5; a = 0;
			break;
			default: // Unrecognised format
				return 0;
		}

		conflist[i++] = EGL_RED_SIZE;
		conflist[i++] = r;

		conflist[i++] = EGL_GREEN_SIZE;
		conflist[i++] = g;

		conflist[i++] = EGL_BLUE_SIZE;
		conflist[i++] = b;

		conflist[i++] = EGL_ALPHA_SIZE;
		conflist[i++] = a;

		// Terminate the list with EGL_NONE
		conflist[i++] = EGL_NONE;

		// Find out how many configs there are in total that match our criteria
		if(!eglChooseConfig(m_EGLDisplay, conflist, NULL, 0, &i32Total_num_configs) || i32Total_num_configs == 0)
			return 0;

		// Allocate an array large enough to store all the possible configs that may be returned
		pConfigs = new EGLConfig[i32Total_num_configs];

		if(!pConfigs)
			return 0;

		// Get all the configs that match our criteria
		if(!eglChooseConfig(m_EGLDisplay, conflist, pConfigs, i32Total_num_configs, &num_config))
		{
			delete pConfigs;
			return 0;
		}

		// Go through the returned configs and try and find a suitable match
		for(j = 0; j < num_config; ++j)
		{
			if((eglGetConfigAttrib(m_EGLDisplay, pConfigs[j], EGL_RED_SIZE,   &value) && value == r)
			&& (eglGetConfigAttrib(m_EGLDisplay, pConfigs[j], EGL_GREEN_SIZE, &value) && value == g)
			&& (eglGetConfigAttrib(m_EGLDisplay, pConfigs[j], EGL_BLUE_SIZE,  &value) && value == b)
			&& (eglGetConfigAttrib(m_EGLDisplay, pConfigs[j], EGL_ALPHA_SIZE, &value) && value == a))
			{
				conf = pConfigs[j];
				break;
			}
		}

		// Tidy up
		delete pConfigs;
	}
	else
#endif
	{
		// Terminate the list with EGL_NONE
		conflist[i++] = EGL_NONE;

		// Return null config if config is not found
		if(!eglChooseConfig(m_EGLDisplay, conflist, &conf, 1, &num_config) || num_config != 1)
		{
			return 0;
		}
	}

	// Return config index
	return conf;
}

/*******************************************************************************
 * Function Name  : StringFrom_eglGetError
 * Returns        : A string
 * Description    : Returns a string representation of an egl error
 *******************************************************************************/
const char *PVRShellInitAPI::StringFrom_eglGetError() const
{
	EGLint nErr = eglGetError();

	switch(nErr)
	{
		case EGL_SUCCESS:
			return "EGL_SUCCESS";
		case EGL_BAD_DISPLAY:
			return "EGL_BAD_DISPLAY";
		case EGL_NOT_INITIALIZED:
			return "EGL_NOT_INITIALIZED";
		case EGL_BAD_ACCESS:
			return "EGL_BAD_ACCESS";
		case EGL_BAD_ALLOC:
			return "EGL_BAD_ALLOC";
		case EGL_BAD_ATTRIBUTE:
			return "EGL_BAD_ATTRIBUTE";
		case EGL_BAD_CONFIG:
			return "EGL_BAD_CONFIG";
		case EGL_BAD_CONTEXT:
			return "EGL_BAD_CONTEXT";
		case EGL_BAD_CURRENT_SURFACE:
			return "EGL_BAD_CURRENT_SURFACE";
		case EGL_BAD_MATCH:
			return "EGL_BAD_MATCH";
		case EGL_BAD_NATIVE_PIXMAP:
			return "EGL_BAD_NATIVE_PIXMAP";
		case EGL_BAD_NATIVE_WINDOW:
			return "EGL_BAD_NATIVE_WINDOW";
		case EGL_BAD_PARAMETER:
			return "EGL_BAD_PARAMETER";
		case EGL_BAD_SURFACE:
			return "EGL_BAD_SURFACE";
		default:
			return "unknown";
	}
}

/*!***********************************************************************
@Function		ApiScreenCaptureBuffer
@Input			Width Width of the region to capture
@Input			Height Height of the region to capture
@Input			pBuf A buffer to put the screen capture into
@description	API-specific function to store the current content of the
				FrameBuffer into the memory allocated by the user.
*************************************************************************/
bool PVRShellInit::ApiScreenCaptureBuffer(int Width,int Height,unsigned char *pBuf)
{
	unsigned char	*pLines2;
	int				i, j;
	bool			bRet = true;


#ifdef BUILD_OVG
	if(eglQueryAPI() != EGL_OPENVG_API)
	{
		m_pShell->PVRShellOutputDebug("PVRShell: Screen capture failed. OpenVG is not the current API");
		return false;
	}
#endif

	/* Allocate memory for line */
	pLines2 = (unsigned char *)calloc(4 * Width * Height, sizeof(unsigned char));
	if (!pLines2) return false;

#ifdef BUILD_OVG
	while (vgGetError());

	vgReadPixels(pLines2, Width * 4, VG_sRGBA_8888, 0, 0, Width, Height);

	if(vgGetError())
	{
		bRet = false;
	}
	else
	{
		/* Convert RGB to BGR in line */
		for (j = 0, i = 0; j < 4 * Width * Height; j += 4, i += 3)
		{
			pBuf[i]   = pLines2[j+1];
			pBuf[i+1] = pLines2[j+2];
			pBuf[i+2] = pLines2[j+3];
		}
	}
#else
	while (glGetError());
	/* Read line from frame buffer */
	glReadPixels(0, 0, Width, Height, GL_RGBA, GL_UNSIGNED_BYTE, pLines2);

	if(glGetError())
	{
		bRet = false;
	}
	else
	{
		/* Convert RGB to BGR in line */
		for (j = 0, i = 0; j < 4 * Width * Height; j += 4, i += 3)
		{
			pBuf[i] = pLines2[j+2];
			pBuf[i+1] = pLines2[j+1];
			pBuf[i+2] = pLines2[j];
		}
	}
#endif

	free(pLines2);
	return bRet;
}

/*!***********************************************************************
 @Function		ApiRenderComplete
 @description	Perform API operations required after a frame has finished (e.g., flipping).
*************************************************************************/
void PVRShellInit::ApiRenderComplete()
{
	bool bRes;

	if(m_pShell->m_pShellData->bNeedPixmap)
	{
		/*
			"Clients rendering to single buffered surfaces (e.g. pixmap surfaces)
			should call eglWaitGL before accessing the native pixmap from the client."
		*/
		eglWaitGL();

		// Pixmap support: Copy the rendered pixmap to the display
		if(m_pShell->m_pShellData->bNeedPixmapDisableCopy)
		{
			bRes = true;
		}
		else
		{
			bRes = OsPixmapCopy();
		}
	}
	else
	{
		if(m_pShell->m_pShellData->bNoShellSwapBuffer)
			return;

		bRes = (eglSwapBuffers (m_EGLDisplay, m_EGLWindow) == EGL_TRUE);
	}

	if(!bRes)
	{
		// check for context loss
#ifdef EGL_VERSION_1_3
		if(eglGetError() == EGL_CONTEXT_LOST)
#else
		if((eglGetError() == EGL_CONTEXT_LOST_IMG) && m_bPowerManagementSupported)
#endif
		{
			m_pShell->ReleaseView();

			OsDoReleaseAPI();
			if(ApiInitAPI())
			{
				m_pShell->InitView();
			}
		}
		else
		{
			m_pShell->PVRShellOutputDebug("eglSwapBuffers failed\n");
		}
	}
}

/*!***********************************************************************
 @Function		ApiSet
 @Input			prefName	Name of value to set
 @Modified		i32Value	Value to set it to
 @description	Set parameters which are specific to the API.
*************************************************************************/
bool PVRShellInit::ApiSet(const prefNameIntEnum prefName, const int i32Value)
{
	switch(prefName)
	{
#ifdef EGL_VERSION_1_1
	case prefSwapInterval:
		m_pShell->m_pShellData->nSwapInterval = i32Value;
		return true;
#endif
#if defined(BUILD_OGLES2) || defined(BUILD_OGLES)
	case prefPriority:
		m_pShell->m_pShellData->nPriority = i32Value;
		return true;
#endif
	case prefConfig:
		m_iRequestedConfig = (EGLint) i32Value;
		return true;

	default:
		return false;
	}
}

/*!***********************************************************************
 @Function		ApiGet
 @Input			prefName	Name of value to get
 @Modified		pn A pointer set to the value asked for
 @description	Get parameters which are specific to the API.
*************************************************************************/
bool PVRShellInit::ApiGet(const prefNameIntEnum prefName, int *pn)
{
	switch(prefName)
	{
		case prefEGLMajorVersion:
			*pn = m_MajorVersion;
			return true;

		case prefEGLMinorVersion:
			*pn = m_MinorVersion;
			return true;

		case prefConfig:
			*pn = (int) (size_t) m_EGLConfig;
			return true;

		default:
			return false;
	}
}

/*!***********************************************************************
 @Function		ApiGet
 @Input			prefName	Name of value to get
 @Modified		pp A pointer set to the value asked for
 @description	Get parameters which are specific to the API.
*************************************************************************/
bool PVRShellInit::ApiGet(const prefNamePtrEnum prefName, void **pp)
{
		switch(prefName)
		{
		case prefEGLDisplay:
			*pp = (void*)m_EGLDisplay;
			return true;

		default:
			return false;
		}
}

/****************************************************************************
** Local code
****************************************************************************/

// The recommended technique for querying OpenGL extensions;
// adapted from http://opengl.org/resources/features/OGLextensions/
static bool PVRShellIsExtensionSupported(EGLDisplay dpy, const char *extension)
{
    const char *extensions = NULL;
    const char *start;
    char *where, *terminator;

    /* Extension names should not have spaces. */
    where = (char *) strchr(extension, ' ');
    if (where || *extension == '\0')
        return false;

    extensions = eglQueryString(dpy, EGL_EXTENSIONS);

    /* It takes a bit of care to be fool-proof about parsing the
    OpenGL extensions string. Don't be fooled by sub-strings, etc. */
    start = extensions;
    for (;;)
	{
        where = (char *) strstr((const char *) start, extension);
        if (!where)
            break;
        terminator = where + strlen(extension);
        if (where == start || *(where - 1) == ' ')
            if (*terminator == ' ' || *terminator == '\0')
                return true;
        start = terminator;
    }
    return false;
}

/*****************************************************************************
 End of file (PVRShellAPI.cpp)
*****************************************************************************/


