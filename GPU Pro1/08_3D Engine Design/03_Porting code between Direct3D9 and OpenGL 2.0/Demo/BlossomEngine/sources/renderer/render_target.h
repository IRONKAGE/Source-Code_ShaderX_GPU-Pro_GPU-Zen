/* $Id: render_target.h 148 2009-08-24 17:28:38Z maxest $ */

#ifndef _BLOSSOM_ENGINE_RENDER_TARGET_
#define _BLOSSOM_ENGINE_RENDER_TARGET_

#ifdef RNDR_D3D
	#include <d3d9.h>
#endif

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CApplication;
	class CRenderer;

	// ----------------------------------------------------------------------------

	enum RenderTargetFormat
	{
		rtfRGBA = 21,
		rtfOne32BitFloatChannel = 114 // for OpenGL it's alpha, for Direct3D it's red
	};

	// ----------------------------------------------------------------------------

	class CRenderTarget
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

		RenderTargetFormat format;
		int width, height;

	public:
		CRenderTarget() { exists = false; }

		void init(RenderTargetFormat format, int width, int height, bool addToRendererRenderTargetsList = true);
		void free();

		RenderTargetFormat getFormat();
		int getWidth();
		int getHeight();
	};
}

// ----------------------------------------------------------------------------

#endif
