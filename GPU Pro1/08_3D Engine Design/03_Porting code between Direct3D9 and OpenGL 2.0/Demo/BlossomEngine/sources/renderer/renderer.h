/* $Id: renderer.h 218 2009-09-03 01:40:14Z maxest $ */

#ifndef _BLOSSOM_ENGINE_RENDERER_
#define _BLOSSOM_ENGINE_RENDERER_

#include <Cg/cg.h>

#ifdef RNDR_D3D
	#include <d3d9.h>
#endif

#include <vector>

#include "../math/blossom_engine_math.h"
#include "texture.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CApplication;
    class CVector3;
	class CVertexDeclaration;
	class CVertexBuffer;
	class CIndexBuffer;
	class CTexture;
	class CRenderTarget;
	class CDepthStencilSurface;
	class CShader;
	class CSpritesManager;
	class CGUIManager;
	class CMesh;
	class CVideoManager;

	// ----------------------------------------------------------------------------

    enum CullMode
    {
        cmNone = 0,
        cmCW = 1,
        cmCCW = 2
    };

	#ifdef RNDR_D3D

		enum TestingFunction
		{
			tfNever = 1,
			tfLess = 2,
			tfEqual = 3,
			tfLessEqual = 4,
			tfGreater = 5,
			tfNotEqual = 6,
			tfGreaterEqual = 7,
			tfAlways = 8
		};

		enum StencilOperation
		{
			soKeep = 1,
			soZero = 2,
			soReplace = 3,
			soIncrease = 7,
			soDecrease = 8
		};

		enum BlendingFunction
		{
			bfZero = 1,
			bfOne = 2,
			bfSrcColor = 3,
			bfInvSrcColor = 4,
			bfSrcAlpha = 5,
			bfInvSrcAlpha = 6,
			bfDestAlpha = 7,
			bfInvDestAlpha = 8,
			bfDestColor = 9,
			bfInvDestColor = 10
		};

		enum PrimitiveType
		{
			ptTriangleList = 4,
			ptTriangleStrip = 5
		};

	#else

		enum TestingFunction
		{
			tfNever =  0x0200,
			tfLess = 0x0201,
			tfEqual = 0x0202,
			tfLessEqual = 0x0203,
			tfGreater = 0x0204,
			tfNotEqual = 0x0205,
			tfGreaterEqual = 0x0206,
			tfAlways = 0x0207
		};

		enum StencilOperation
		{
			soKeep = 0x1E00,
			soZero = 0,
			soReplace = 0x1E01,
			soIncrease = 0x8507,
			soDecrease = 0x8508
		};

		enum BlendingFunction
		{
			bfZero = 0,
			bfOne = 1,
			bfSrcColor = 0x0300,
			bfInvSrcColor = 0x0301,
			bfSrcAlpha = 0x0302,
			bfInvSrcAlpha = 0x0303,
			bfDestAlpha = 0x0304,
			bfInvDestAlpha = 0x0305,
			bfDestColor = 0x0306,
			bfInvDestColor = 0x0307
		};

		enum PrimitiveType
		{
			ptTriangleList = 4,
			ptTriangleStrip = 5
		};

	#endif

	// ----------------------------------------------------------------------------

    class CRenderer
    {
		friend class CApplication;
		friend class CVertexDeclaration;
		friend class CVertexBuffer;
		friend class CIndexBuffer;
		friend class CTexture;
		friend class CRenderTarget;
		friend class CDepthStencilSurface;
		friend class CShader;
		friend class CSpritesManager;
		friend class CGUIManager;
		friend class CMesh;
		friend class CVideoManager;

	private:
		#ifdef RNDR_D3D
			static IDirect3D9 *D3DObject;
			static IDirect3DDevice9 *D3DDevice;

			static D3DPRESENT_PARAMETERS presentParameters;

			static IDirect3DSurface9 *backBufferTargetSurface;
			static IDirect3DSurface9 *backBufferDepthStencilSurface;

			static std::vector<CRenderTarget*> renderTargets;
			static std::vector<CDepthStencilSurface*> depthStencilSurfaces;
		#else
			static unsigned int offScreenFramebuffer;
		#endif

		static CGcontext cgContext;
		static bool isNVidiaGPU;

		static bool vertexDeclarationChanged;

		static CullMode currentCullMode;
		static CRenderTarget *currentRenderTarget;
		static CDepthStencilSurface *currentDepthStencilSurface;
		static CVertexDeclaration *currentVertexDeclaration;
		static CVertexBuffer *currentVertexBuffer;
		static CIndexBuffer *currentIndexBuffer;
		static CShader *currentVertexShader;
		static CShader *currentPixelShader;
		static CTexture *currentSamplerTexture[4];
		static TextureFiltering currentSamplerMagFiltering[4];
		static TextureFiltering currentSamplerMinFiltering[4];
		static TextureFiltering currentSamplerMipFiltering[4];
		static TextureAddressing currentSamplerAddressing[4];
		static CVector3 currentSamplerBorderColor[4];

		// these are default vertex declarations and shaders that might be useful for experimentation
		static CVertexDeclaration meshVertexDeclaration;
		static CVertexDeclaration spriteVertexDeclaration;
		static CVertexDeclaration guiEntityVertexDeclaration;
		static CShader meshStaticVertexShader, meshAnimationVertexShader, meshAnimationInterpolationVertexShader;
		static CShader spriteVertexShader;
		static CShader guiEntityVertexShader;
		static CShader meshPixelShader;
		static CShader spritePixelShader;
		static CShader guiEntityPixelShader;

	public:
		static void init();
		static void free();

		static void clear(bool target, bool depth, bool stencil, const CVector3 &targetColor);

		static void setViewport(int x, int y, int width, int height); // (x, y) is the viewport's center
		static void setDepthRange(float minZ, float maxZ);
		static void setCullMode(CullMode cullMode);
		static void setScissorTestState(bool state);
		static void setScissorTestRect(int x, int y, int width, int height); // (x, y) is the scissor rect's center

		static void setTargetWriteState(bool state);

		static void setDepthWriteState(bool state);
		static void setDepthTestingFunction(TestingFunction testingFunction);

		static void setStencilState(bool state);
		static void setStencilTwoSidedState(bool state);
		static void setStencilFunction(TestingFunction testingFunction, unsigned int reference, unsigned int mask);
		static void setStencilOperation(StencilOperation fail, StencilOperation zfail, StencilOperation pass);
		static void setStencilOperationBackFace(StencilOperation fail, StencilOperation zfail, StencilOperation pass);
		static void setStencilMask(unsigned int mask);

		static void setBlendingState(bool state);
		static void setBlendingFunction(BlendingFunction srcBlendingFunction, BlendingFunction destBlendingFunction);

		static void setRenderTarget(const CRenderTarget *renderTarget);
		static void setDepthStencilSurface(const CDepthStencilSurface *depthStencilSurface);
		static void setVertexDeclaration(const CVertexDeclaration &vertexDeclaration);
		static void setVertexBuffer(const CVertexBuffer &vertexBuffer);
		static void setIndexBuffer(const CIndexBuffer &indexBuffer);
		static void setShader(const CShader &shader);
		static void setTexture(int sampler, const CTexture &texture);
		static void setTexture(int sampler, const CRenderTarget &renderTarget);
		static void setTexture(int sampler, const CDepthStencilSurface &depthStencilSurface);

		static void drawPrimitives(PrimitiveType primitiveType, int startVertex, int primitivesNum);
		static void drawIndexedPrimitives(PrimitiveType primitiveType, int verticesNum, int primitivesNum);
    };
}

// ----------------------------------------------------------------------------

#endif
