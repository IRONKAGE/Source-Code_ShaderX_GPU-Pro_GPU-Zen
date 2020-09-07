/* $Id: depthstencil_surface.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#ifdef RNDR_D3D
	#include <d3d9.h>
#else
	#include <GL/glew.h>
	#include <Cg/cgGL.h>
#endif

#include "depthstencil_surface.h"
#include "renderer.h"
#include "texture.h"
#include "../math/blossom_engine_math.h"
#include "../common/blossom_engine_common.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	int CDepthStencilSurface::referenceCounter = 0;



	void CDepthStencilSurface::init(DepthStencilSurfaceFormat format, int width, int height, bool addToRendererDepthStencilSurfacesList)
	{
		if (exists)
			return;

		this->format = format;
		this->width = width;
		this->height = height;

		#ifdef RNDR_D3D
		{	
			if (format == dssfShadowMap)
			{
				CRenderer::D3DDevice->CreateTexture(width, height, 1, D3DUSAGE_DEPTHSTENCIL, D3DFMT_D24X8, D3DPOOL_DEFAULT, &texture, NULL);
				texture->GetSurfaceLevel(0, &surface);
			}
			else
			{
				CRenderer::D3DDevice->CreateDepthStencilSurface(width, height, D3DFORMAT(format), D3DMULTISAMPLE_NONE, 0, false, &surface, NULL);
			}

			if (addToRendererDepthStencilSurfacesList)
				CRenderer::depthStencilSurfaces.push_back(this);
		}
		#else
		{
			glGenTextures(1, &texture);
			glActiveTexture(GL_TEXTURE0 + 4);
			glBindTexture(GL_TEXTURE_2D, texture);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

			if (format != dssfShadowMap)
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			}
			else
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

				float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
				glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
			}

			if (format == dssfD24X8 || format == dssfShadowMap)
				glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
			else
				glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL);
		}
		#endif

		exists = true;
		referenceCounter++;
		CLogger::addText("\tOK: depthstencil surface created (reference counter = %d)\n", referenceCounter);
	}



	void CDepthStencilSurface::free()
	{
		if (!exists)
			return;

		#ifdef RNDR_D3D
		{
			surface->Release();
			if (format == dssfShadowMap)
				texture->Release();
		}
		#else
		{
			glDeleteTextures(1, &texture);
		}
		#endif

		exists = false;
		referenceCounter--;
		CLogger::addText("\tOK: depthstencil surface freed (reference counter = %d)\n", referenceCounter);
	}



	DepthStencilSurfaceFormat CDepthStencilSurface::getFormat()
	{
		return format;
	}



	int CDepthStencilSurface::getWidth()
	{
		return width;
	}



	int CDepthStencilSurface::getHeight()
	{
		return height;
	}
}
