/* $Id: depthstencil_surface.h 148 2009-08-24 17:28:38Z maxest $ */

#ifndef _BLOSSOM_ENGINE_DEPTHSTENCIL_SURFACE_
#define _BLOSSOM_ENGINE_DEPTHSTENCIL_SURFACE_

#ifdef RNDR_D3D
	#include <d3d9.h>
#endif

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CApplication;
	class CRenderer;

	// ----------------------------------------------------------------------------

	enum DepthStencilSurfaceFormat
	{
		dssfD24X8 = 77,
		dssfShadowMap = 76,
		dssfD24S8 = 75
	};

	// ----------------------------------------------------------------------------

	class CDepthStencilSurface
	{
		friend class CApplication;
		friend class CRenderer;

	private:
		static int referenceCounter;
		bool exists;

		#ifdef RNDR_D3D
			IDirect3DTexture9 *texture;
			IDirect3DSurface9 *surface;
		#else
			unsigned int texture;
		#endif

		DepthStencilSurfaceFormat format;
		int width, height;

	public:
		CDepthStencilSurface() { exists = false; }

		void init(DepthStencilSurfaceFormat format, int width, int height, bool addToRendererDepthStencilSurfacesList = true);
		void free();

		DepthStencilSurfaceFormat getFormat();
		int getWidth();
		int getHeight();
	};
}

// ----------------------------------------------------------------------------

#endif
