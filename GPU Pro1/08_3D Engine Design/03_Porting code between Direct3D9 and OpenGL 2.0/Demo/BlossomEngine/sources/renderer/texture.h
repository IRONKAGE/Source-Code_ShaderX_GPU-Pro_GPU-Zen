/* $Id: texture.h 183 2009-08-27 17:10:14Z maxest $ */

#ifndef _BLOSSOM_ENGINE_TEXTURE_
#define _BLOSSOM_ENGINE_TEXTURE_

#include <string>

#ifdef RNDR_D3D
	#include <d3d9.h>
#endif

#include "../math/blossom_engine_math.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CRenderer;
	class CVector3;

	// ----------------------------------------------------------------------------

    enum TextureFiltering
    {
		tfNone = 0,
        tfPoint = 1,
        tfLinear = 2
    };

	#ifdef RNDR_D3D
		enum TextureAddressing
		{
			taWrap = 1,
			taClamp = 3,
			taBorder = 4
		};
	#else
		enum TextureAddressing
		{
			taWrap = 0x2901,
			taClamp = 0x812F,
			taBorder = 0x812D
		};
	#endif

	// ----------------------------------------------------------------------------

	class CTexture
	{
		friend class CRenderer;

	private:
		static int referenceCounter;
		bool exists;

		#ifdef RNDR_D3D
			IDirect3DTexture9 *id;
		#else
			unsigned int id;
		#endif

		TextureFiltering magFiltering, minFiltering, mipFiltering;
		TextureAddressing addressing;
		CVector3 borderColor;

	public:
		CTexture() { exists = false; }

		void init();
		void free();

		void loadDataFromFile(std::string fileName, int mipmapsNum, bool compress = false);

		void setTextureFiltering(TextureFiltering mag, TextureFiltering min, TextureFiltering mip);
		void setTextureAddressing(TextureAddressing textureAddressing);
		void setTextureBorderColor(const CVector3 &color);
	};
}

// ----------------------------------------------------------------------------

#endif
