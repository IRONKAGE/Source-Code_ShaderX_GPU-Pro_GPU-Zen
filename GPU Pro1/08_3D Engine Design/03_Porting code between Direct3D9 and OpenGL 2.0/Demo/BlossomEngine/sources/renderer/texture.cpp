/* $Id: texture.cpp 256 2009-09-08 17:34:29Z maxest $ */

#ifdef RNDR_D3D
	#include <d3d9.h>
	#include <d3dx9.h>
#else
	#include <SDL/SDL_image.h>
	#include <GL/glew.h>
#endif

#include "texture.h"
#include "renderer.h"
#include "../math/blossom_engine_math.h"
#include "../common/blossom_engine_common.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	int CTexture::referenceCounter = 0;



	void CTexture::init()
	{
		if (exists)
			return;

		this->magFiltering = tfLinear;
		this->minFiltering = tfLinear;
		this->mipFiltering = tfNone;

		this->addressing = taWrap;
		this->borderColor = CVector3();

		#ifdef RNDR_D3D
		{
			id = NULL;
		}
		#else
		{
			glGenTextures(1, &id);
			glActiveTexture(GL_TEXTURE0 + 4);
			glBindTexture(GL_TEXTURE_2D, id);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		}
		#endif

		exists = true;
		referenceCounter++;
		CLogger::addText("\tOK: texture created (reference counter = %d)\n", referenceCounter);
	}



	void CTexture::free()
	{
		if (!exists)
			return;

		#ifdef RNDR_D3D
		{
			if (id != NULL)
			{
				id->Release();
				id = NULL;
			}
		}
		#else
		{
			glDeleteTextures(1, &id);
		}
		#endif

		exists = false;
		referenceCounter--;
		CLogger::addText("\tOK: texture freed (reference counter = %d)\n", referenceCounter);
	}



	void CTexture::loadDataFromFile(std::string fileName, int mipmapsNum, bool compress)
	{
		#ifdef RNDR_D3D
		{
			if (FAILED(D3DXCreateTextureFromFileEx(CRenderer::D3DDevice, fileName.c_str(), D3DX_DEFAULT, D3DX_DEFAULT, mipmapsNum, 0, D3DFMT_UNKNOWN, D3DPOOL_MANAGED, D3DX_DEFAULT, D3DX_DEFAULT, 0, NULL, NULL, &id)))
			{
				CLogger::addText("\tERROR: couldn't load texture from file, %s\n", fileName.c_str());
				exit(1);
			}

			if (compress)
			{
				D3DFORMAT format;
				int width, height;

				D3DSURFACE_DESC surfaceDescription;
				id->GetLevelDesc(0, &surfaceDescription);

				format = surfaceDescription.Format;
				width = surfaceDescription.Width;
				height = surfaceDescription.Height;

				// if loaded texture isn't already compressed, then do the compression
				if (format != D3DFMT_DXT1 && format != D3DFMT_DXT2 && format != D3DFMT_DXT3 && format != D3DFMT_DXT4 && format != D3DFMT_DXT5)
				{
					if (mipmapsNum == 0)
						mipmapsNum = id->GetLevelCount();

					if (format == D3DFMT_A8R8G8B8)
						format = D3DFMT_DXT5;
					else
						format = D3DFMT_DXT1;

					id->Release();
					CRenderer::D3DDevice->CreateTexture(width, height, mipmapsNum, 0, format, D3DPOOL_MANAGED, &id, NULL);
					
					IDirect3DSurface9 *surface;
					for (int i = 0; i < mipmapsNum; i++)
					{
						id->GetSurfaceLevel(i, &surface);
						D3DXLoadSurfaceFromFile(surface, NULL, NULL, fileName.c_str(), NULL, D3DX_DEFAULT, 0, NULL);
						surface->Release();
					}
				}
			}

			this->magFiltering = tfLinear;
			this->minFiltering = tfLinear;
			if (mipmapsNum == 1)
				this->mipFiltering = tfNone;
			else
				this->mipFiltering = tfLinear;
		}
		#else
		{
			glActiveTexture(GL_TEXTURE0 + 4);
			glBindTexture(GL_TEXTURE_2D, id);

			if (mipmapsNum == 1)
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
				glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, false);

				setTextureFiltering(tfLinear, tfLinear, tfNone);
			}
			else
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, mipmapsNum - 1);
				glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, true);

				setTextureFiltering(tfLinear, tfLinear, tfLinear);
			}

			SDL_Surface *surface = IMG_Load(fileName.c_str());

			if (surface == NULL)
			{
				CLogger::addText("\tERROR: couldn't load texture from file, %s\n", fileName.c_str());
				exit(1);
			}

			unsigned int internalFormat;

			if (surface->format->Amask)
			{
				if (compress)
					internalFormat = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
				else
					internalFormat = GL_RGBA;

				glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, surface->w, surface->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, surface->pixels);
			}
			else
			{
				if (compress)
					internalFormat = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
				else
					internalFormat = GL_RGB;

				glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, surface->w, surface->h, 0, GL_RGB, GL_UNSIGNED_BYTE, surface->pixels);
			}

			SDL_FreeSurface(surface);
		}
		#endif

		CLogger::addText("\tOK: texture loaded from file, %s\n", fileName.c_str());
	}



	void CTexture::setTextureFiltering(TextureFiltering mag, TextureFiltering min, TextureFiltering mip)
	{
		magFiltering = mag;
		minFiltering = min;
		mipFiltering = mip;

		#ifdef RNDR_D3D
		{
			// nothing to be done
		}
		#else
		{
			glActiveTexture(GL_TEXTURE0 + 4);
			glBindTexture(GL_TEXTURE_2D, id);

			{
				if (mag == tfPoint)
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				else if (mag == tfLinear)
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			}

			{
				if (min == tfPoint && mip == tfNone)
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				else if (min == tfLinear && mip == tfNone)
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

				else if (min == tfPoint && mip == tfPoint)
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
				else if (min == tfLinear && mip == tfPoint)
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);

				else if (min == tfPoint && mip == tfLinear)
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
				else if (min == tfLinear && mip == tfLinear)
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			}
		}
		#endif
	}



	void CTexture::setTextureAddressing(TextureAddressing textureAddressing)
	{
		this->addressing = textureAddressing;

		#ifdef RNDR_D3D
		{
			// nothing to be done
		}
		#else
		{
			glActiveTexture(GL_TEXTURE0 + 4);
			glBindTexture(GL_TEXTURE_2D, id);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, textureAddressing);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, textureAddressing);
		}
		#endif
	}



	void CTexture::setTextureBorderColor(const CVector3 &color)
	{
		this->borderColor = color;

		#ifdef RNDR_D3D
		{
			// nothing to be done
		}
		#else
		{
			glActiveTexture(GL_TEXTURE0 + 4);
			glBindTexture(GL_TEXTURE_2D, id);

			float borderColor[] = { color.x, color.y, color.z, 1.0f };
			glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
		}
		#endif
	}
}
