/* $Id: render_target.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#ifdef RNDR_D3D
	#include <d3d9.h>
#else
	#include <GL/glew.h>
	#include <Cg/cgGL.h>
#endif

#include "render_target.h"
#include "renderer.h"
#include "texture.h"
#include "../math/blossom_engine_math.h"
#include "../common/blossom_engine_common.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	int CRenderTarget::referenceCounter = 0;



	void CRenderTarget::init(RenderTargetFormat format, int width, int height, bool addToRendererRenderTargetsList)
	{
		if (exists)
			return;

		this->format = format;
		this->width = width;
		this->height = height;

		#ifdef RNDR_D3D
		{
			CRenderer::D3DDevice->CreateTexture(width, height, 1, D3DUSAGE_RENDERTARGET, D3DFORMAT(format), D3DPOOL_DEFAULT, &texture, NULL);
			texture->GetSurfaceLevel(0, &surface);

			if (addToRendererRenderTargetsList)
				CRenderer::renderTargets.push_back(this);
		}
		#else
		{
			glGenTextures(1, &texture);
			glActiveTexture(GL_TEXTURE0 + 4);
			glBindTexture(GL_TEXTURE_2D, texture);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

			if (format == rtfRGBA)
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			else
				glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA32F_ARB, width, height, 0, GL_ALPHA, GL_FLOAT, NULL);
		}
		#endif

		exists = true;
		referenceCounter++;
		CLogger::addText("\tOK: render target created (reference counter = %d)\n", referenceCounter);
	}



	void CRenderTarget::free()
	{
		if (!exists)
			return;

		#ifdef RNDR_D3D
		{
			surface->Release();
			texture->Release();
		}
		#else
		{
			glDeleteTextures(1, &texture);
		}
		#endif

		exists = false;
		referenceCounter--;
		CLogger::addText("\tOK: render target freed (reference counter = %d)\n", referenceCounter);
	}



	RenderTargetFormat CRenderTarget::getFormat()
	{
		return format;
	}



	int CRenderTarget::getWidth()
	{
		return width;
	}



	int CRenderTarget::getHeight()
	{
		return height;
	}
}
