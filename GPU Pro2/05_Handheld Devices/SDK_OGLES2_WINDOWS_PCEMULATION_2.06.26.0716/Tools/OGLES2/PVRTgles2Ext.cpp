/******************************************************************************

 @File         PVRTgles2Ext.cpp

 @Title        PVRTgles2Ext

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  OpenGL ES 2.0 extensions

******************************************************************************/
#include <string.h>

#include "PVRTContext.h"
#include "PVRTgles2Ext.h"

/****************************************************************************
** Local code
****************************************************************************/
#if defined(TARGET_OS_IPHONE)
// the extensions supported on the iPhone are treated as core functions of gl
// so use this macro to assign the function pointers in this class appropriately.
#define PVRGetProcAddress(x) ::x
#else

#if defined(EGL_NOT_PRESENT) || (defined(__BADA__) && defined(_X86_)) // Bada simulator

#if defined(__PALMPDK__)
#include "SDL.h"

#define PVRGetProcAddress(x) SDL_GLES_GetProcAddress(#x)
#else
#define PVRGetProcAddress(x) NULL
#endif

#else
#define PVRGetProcAddress(x) eglGetProcAddress(#x)
#endif

#endif

/****************************************************************************
** Class: CPVRTgles2Ext
****************************************************************************/

/*!***************************************************************************
 @Function			LoadExtensions
 @Description		Initialises IMG extensions
*****************************************************************************/
void CPVRTgles2Ext::LoadExtensions()
{
	glMultiDrawElementsEXT = 0;
	glMultiDrawArraysEXT = 0;
	glMapBufferOES = 0;
	glUnmapBufferOES = 0;
	glGetBufferPointervOES = 0;

	// Supported extensions provide new entry points for OpenGL ES 2.0.

	const GLubyte *pszGLExtensions;

	/* Retrieve GL extension string */
    pszGLExtensions = glGetString(GL_EXTENSIONS);

#ifndef TARGET_OS_IPHONE
	/* GL_EXT_multi_draw_arrays */
	if (strstr((char *)pszGLExtensions, "GL_EXT_multi_draw_arrays"))
	{
		glMultiDrawElementsEXT = (PFNGLMULTIDRAWELEMENTS) PVRGetProcAddress(glMultiDrawElementsEXT);
		glMultiDrawArraysEXT = (PFNGLMULTIDRAWARRAYS) PVRGetProcAddress(glMultiDrawArraysEXT);
	}

	/* GL_EXT_multi_draw_arrays */
	if (strstr((char *)pszGLExtensions, "GL_OES_mapbuffer"))
	{
        glMapBufferOES = (PFNGLMAPBUFFEROES) PVRGetProcAddress(glMapBufferOES);
        glUnmapBufferOES = (PFNGLUNMAPBUFFEROES) PVRGetProcAddress(glUnmapBufferOES);
        glGetBufferPointervOES = (PFNGLGETBUFFERPOINTERVOES) PVRGetProcAddress(glGetBufferPointervOES);
	}
#endif

}

/*!***********************************************************************
@Function			IsGLExtensionSupported
@Input				extension extension to query for
@Returns			True if the extension is supported
@Description		Queries for support of an extension
*************************************************************************/
bool CPVRTgles2Ext::IsGLExtensionSupported(const char *extension)
{
	// The recommended technique for querying OpenGL extensions;
	// from http://opengl.org/resources/features/OGLextensions/
	const GLubyte *extensions = NULL;
	const GLubyte *start;
	GLubyte *where, *terminator;

	/* Extension names should not have spaces. */
	where = (GLubyte *) strchr(extension, ' ');
	if (where || *extension == '\0')
		return 0;

	extensions = glGetString(GL_EXTENSIONS);

	/* It takes a bit of care to be fool-proof about parsing the
	OpenGL extensions string. Don't be fooled by sub-strings, etc. */
	start = extensions;
	for (;;) {
		where = (GLubyte *) strstr((const char *) start, extension);
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
 End of file (PVRTglesExt.cpp)
*****************************************************************************/

