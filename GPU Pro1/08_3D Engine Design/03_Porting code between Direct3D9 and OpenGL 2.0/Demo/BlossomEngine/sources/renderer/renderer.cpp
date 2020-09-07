/* $Id: renderer.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#include <Cg/cg.h>

#ifdef RNDR_D3D
	#include <d3d9.h>
	#include <Cg/cgD3D9.h>
#else
	#include <GL/glew.h>
	#include <Cg/cgGL.h>
#endif

#include <string.h>

#include "renderer.h"
#include "vertex_declaration.h"
#include "vertex_buffer.h"
#include "index_buffer.h"
#include "texture.h"
#include "render_target.h"
#include "depthstencil_surface.h"
#include "shader.h"
#include "../math/blossom_engine_math.h"
#include "../common/blossom_engine_common.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	#ifdef RNDR_D3D
		IDirect3D9 *CRenderer::D3DObject;
		IDirect3DDevice9 *CRenderer::D3DDevice;

		D3DPRESENT_PARAMETERS CRenderer::presentParameters;

		IDirect3DSurface9 *CRenderer::backBufferTargetSurface;
		IDirect3DSurface9 *CRenderer::backBufferDepthStencilSurface;

		std::vector<CRenderTarget*> CRenderer::renderTargets;
		std::vector<CDepthStencilSurface*> CRenderer::depthStencilSurfaces;
	#else
		unsigned int CRenderer::offScreenFramebuffer;
	#endif

	CGcontext CRenderer::cgContext;
	bool CRenderer::isNVidiaGPU;

	bool CRenderer::vertexDeclarationChanged;

	CullMode CRenderer::currentCullMode;
	CRenderTarget *CRenderer::currentRenderTarget;
	CDepthStencilSurface *CRenderer::currentDepthStencilSurface;
	CVertexDeclaration *CRenderer::currentVertexDeclaration;
	CVertexBuffer *CRenderer::currentVertexBuffer;
	CIndexBuffer *CRenderer::currentIndexBuffer;
	CShader *CRenderer::currentVertexShader;
	CShader *CRenderer::currentPixelShader;
	CTexture *CRenderer::currentSamplerTexture[4];
	TextureFiltering CRenderer::currentSamplerMagFiltering[4];
	TextureFiltering CRenderer::currentSamplerMinFiltering[4];
	TextureFiltering CRenderer::currentSamplerMipFiltering[4];
	TextureAddressing CRenderer::currentSamplerAddressing[4];
	CVector3 CRenderer::currentSamplerBorderColor[4];

	CVertexDeclaration CRenderer::meshVertexDeclaration;
	CVertexDeclaration CRenderer::spriteVertexDeclaration;
	CVertexDeclaration CRenderer::guiEntityVertexDeclaration;
	CShader CRenderer::meshStaticVertexShader, CRenderer::meshAnimationVertexShader, CRenderer::meshAnimationInterpolationVertexShader;
	CShader CRenderer::spriteVertexShader;
	CShader CRenderer::guiEntityVertexShader;
	CShader CRenderer::meshPixelShader;
	CShader CRenderer::spritePixelShader;
	CShader CRenderer::guiEntityPixelShader;
	


    void CRenderer::init()
    {
		cgContext = cgCreateContext();

		#ifdef RNDR_D3D
		{
			D3DDevice->GetRenderTarget(0, &backBufferTargetSurface);
			D3DDevice->GetDepthStencilSurface(&backBufferDepthStencilSurface);

			D3DDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
			D3DDevice->SetRenderState(D3DRS_ZFUNC, D3DCMP_LESSEQUAL);

			cgD3D9SetDevice(D3DDevice);

			D3DADAPTER_IDENTIFIER9 adapterIdentifier;
			D3DObject->GetAdapterIdentifier(D3DADAPTER_DEFAULT, 0, &adapterIdentifier);

			if (strstr(adapterIdentifier.Description, "NVIDIA"))
				isNVidiaGPU = true;
			else
				isNVidiaGPU = false;

			CLogger::addText("OK: Renderer start, Direct3D9, %s\n", adapterIdentifier.Description);
		}
		#else
		{
			glewInit();

			glEnable(GL_DEPTH_TEST);
			glEnable(GL_CULL_FACE);
			glCullFace(GL_BACK);
			glFrontFace(GL_CCW);
			glDepthFunc(GL_LEQUAL);

			glGenFramebuffersEXT(1, &offScreenFramebuffer);

			const char *vendor = (const char*)glGetString(GL_VENDOR);

			if (strstr(vendor, "NVIDIA"))
				isNVidiaGPU = true;
			else
				isNVidiaGPU = false;

			if (isNVidiaGPU)
			{
				cgGLEnableProfile(CG_PROFILE_VP40);
				cgGLEnableProfile(CG_PROFILE_FP40);
			}
			else
			{
				cgGLEnableProfile(CG_PROFILE_ARBVP1);
				cgGLEnableProfile(CG_PROFILE_ARBFP1);
			}

			CLogger::addText("OK: Renderer start, OpenGL %s, %s, %s\n", (const char*)glGetString(GL_VERSION), (const char*)glGetString(GL_VENDOR), (const char*)glGetString(GL_RENDERER));
		}
		#endif

		vertexDeclarationChanged = false;

        currentCullMode = cmCW;
		currentRenderTarget = NULL;
		currentDepthStencilSurface = NULL;
		currentVertexDeclaration = NULL;
		currentVertexBuffer = NULL;
		currentIndexBuffer = NULL;
		currentVertexShader = NULL;
		currentPixelShader = NULL;
		for (int i = 0; i < 4; i++)
		{
			currentSamplerTexture[i] = NULL;
			currentSamplerMagFiltering[i] = (TextureFiltering)-1;
			currentSamplerMinFiltering[i] = (TextureFiltering)-1;
			currentSamplerMipFiltering[i] = (TextureFiltering)-1;
			currentSamplerAddressing[i] = (TextureAddressing)-1;
			currentSamplerBorderColor[i] = CVector3(-1.0f, -1.0f, -1.0f);
		}

		meshVertexDeclaration.init(3, 0, 3, 3, 3, 2, 1);
		spriteVertexDeclaration.init(3, 4, 0, 2, 3, 3, 3, 3);
		guiEntityVertexDeclaration.init(2, 4, 0, 2);

		meshStaticVertexShader.init("./BlossomEngine/shaders/renderer/mesh_static.vs", stVertexShader);
		meshAnimationVertexShader.init("./BlossomEngine/shaders/renderer/mesh_animation.vs", stVertexShader);
		meshAnimationInterpolationVertexShader.init("./BlossomEngine/shaders/renderer/mesh_animation_interpolation.vs", stVertexShader);
		spriteVertexShader.init("./BlossomEngine/shaders/renderer/sprite.vs", stVertexShader);
		guiEntityVertexShader.init("./BlossomEngine/shaders/renderer/gui_entity.vs", stVertexShader);
		
		meshPixelShader.init("./BlossomEngine/shaders/renderer/mesh.ps", stPixelShader);
		spritePixelShader.init("./BlossomEngine/shaders/renderer/sprite.ps", stPixelShader);
		guiEntityPixelShader.init("./BlossomEngine/shaders/renderer/gui_entity.ps", stPixelShader);
    }



	void CRenderer::free()
	{
		meshVertexDeclaration.free();
		spriteVertexDeclaration.free();
		guiEntityVertexDeclaration.free();

		meshStaticVertexShader.free();
		meshAnimationVertexShader.free();
		meshAnimationInterpolationVertexShader.free();
		spriteVertexShader.free();
		guiEntityVertexShader.free();

		meshPixelShader.free();
		spritePixelShader.free();
		guiEntityPixelShader.free();

		#ifdef RNDR_D3D
		{
			cgD3D9SetDevice(0);

			renderTargets.clear();

			if (backBufferDepthStencilSurface != NULL)
			{
				backBufferDepthStencilSurface->Release();
				backBufferDepthStencilSurface = NULL;
			}

			if (backBufferTargetSurface != NULL)
			{
				backBufferTargetSurface->Release();
				backBufferTargetSurface = NULL;
			}

			if (D3DDevice != NULL)
			{
				D3DDevice->Release();
				D3DDevice = NULL;
			}

			if (D3DObject != NULL)
			{
				D3DObject->Release();
				D3DObject = NULL;
			}
		}
		#else
		{
			glDeleteFramebuffersEXT(1, &offScreenFramebuffer);

			if (isNVidiaGPU)
			{
				cgGLDisableProfile(CG_PROFILE_VP40);
				cgGLDisableProfile(CG_PROFILE_FP40);
			}
			else
			{
				cgGLDisableProfile(CG_PROFILE_ARBVP1);
				cgGLDisableProfile(CG_PROFILE_ARBFP1);
			}
		}
		#endif

		cgDestroyContext(cgContext);

		CLogger::addText("OK: Renderer quit\n");
		CLogger::addText("\tRender targets reference counter: %d\n", CRenderTarget::referenceCounter);
		CLogger::addText("\tDepthStencil surfaces reference counter: %d\n", CDepthStencilSurface::referenceCounter);
		CLogger::addText("\tVertex declarations reference counter: %d\n", CVertexDeclaration::referenceCounter);
		CLogger::addText("\tVertex buffers reference counter: %d\n", CVertexBuffer::referenceCounter);
		CLogger::addText("\tShaders reference counter: %d\n", CShader::referenceCounter);
		CLogger::addText("\tTextures reference counter: %d\n", CTexture::referenceCounter);
	}



    void CRenderer::clear(bool target, bool depth, bool stencil, const CVector3 &targetColor)
    {
		unsigned int flags = 0;

		#ifdef RNDR_D3D
		{
			if (target)
				flags |= D3DCLEAR_TARGET;
			if (depth)
				flags |= D3DCLEAR_ZBUFFER;
			if (stencil)
				flags |= D3DCLEAR_STENCIL;

			DWORD color = D3DCOLOR_XRGB((int)(255.0f*targetColor.x), (int)(255.0f*targetColor.y), (int)(255.0f*targetColor.z));
			D3DDevice->Clear(0, NULL, flags, color, 1.0f, 0);
		}
		#else
		{
			if (target)
			{
				glClearColor(targetColor.x, targetColor.y, targetColor.z, 1.0f);
				flags |= GL_COLOR_BUFFER_BIT;
			}
			if (depth)
			{
				glClearDepth(1.0f);
				flags |= GL_DEPTH_BUFFER_BIT;
			}
			if (stencil)
			{
				glClearStencil(0);
				flags |= GL_STENCIL_BUFFER_BIT;
			}

			glClear(flags);
		}
		#endif
    }



	void CRenderer::setViewport(int x, int y, int width, int height) // (x, y) is the scissor rect's center
	{
		int left = x - width/2;
		int top = y - height/2;
		int right = x + width/2;
		int bottom = y + height/2;

		#ifdef RNDR_D3D
		{
			RECT scissorRect = { left, top, right, bottom };
			D3DDevice->SetScissorRect(&scissorRect);

			D3DVIEWPORT9 viewport;
			D3DDevice->GetViewport(&viewport);

			viewport.X = left;
			viewport.Y = top;
			viewport.Width = right - left;
			viewport.Height = bottom - top;

			D3DDevice->SetViewport(&viewport);
		}
		#else
		{
			glViewport(left, CApplication::getScreenHeight() - bottom, right - left, bottom - top);
		}
		#endif
	}



	void CRenderer::setDepthRange(float minZ, float maxZ)
	{
		#ifdef RNDR_D3D
		{
			D3DVIEWPORT9 viewport;
			D3DDevice->GetViewport(&viewport);

			viewport.MinZ = minZ;
			viewport.MaxZ = maxZ;

			D3DDevice->SetViewport(&viewport);
		}
		#else
		{
			glDepthRange(minZ, maxZ);
		}
		#endif
	}



    void CRenderer::setCullMode(CullMode cullMode)
    {
		if (cullMode == currentCullMode)
			return;

		currentCullMode = cullMode;

        switch (cullMode)
        {
			#ifdef RNDR_D3D
			{
				case cmNone:
					D3DDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
					break;

				case cmCW:
					D3DDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
					break;

				case cmCCW:
					D3DDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
					break;				
			}
			#else
			{
				case cmNone:
					glDisable(GL_CULL_FACE);
					break;

				case cmCW:
					glEnable(GL_CULL_FACE);
					glFrontFace(GL_CCW);
					break;

				case cmCCW:
					glEnable(GL_CULL_FACE);
					glFrontFace(GL_CW);
					break;
			}
			#endif
        }
    }



	void CRenderer::setScissorTestState(bool state)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetRenderState(D3DRS_SCISSORTESTENABLE, state);		
		}
		#else
		{
			if (state)
				glEnable(GL_SCISSOR_TEST);
			else
				glDisable(GL_SCISSOR_TEST);
		}
		#endif
	}



	void CRenderer::setScissorTestRect(int x, int y, int width, int height)
	{
		int screenWidth = CApplication::getScreenWidth();
		int screenHeight = CApplication::getScreenHeight();
		int left, top, right, bottom;

		left = x - width/2;
		top = y - height/2;
		right = x + width/2;
		bottom = y + height/2;

		if (left < 0)
			left = 0;
		if (top < 0)
			top = 0;
		if (right < 0)
			right = 0;
		if (bottom < 0)
			bottom = 0;
		if (left > screenWidth)
			left = screenWidth;
		if (top > screenHeight)
			top = screenHeight;
		if (right > screenWidth)
			right = screenWidth;
		if (bottom > screenHeight)
			bottom = screenHeight;
		if (right < left)
		{
			left = 0;
			right = screenWidth;
		}
		if (bottom < top)
		{
			top = 0;
			bottom = screenHeight;
		}

		#ifdef RNDR_D3D
		{
			RECT scissorRect = { left, top, right, bottom };
			D3DDevice->SetScissorRect(&scissorRect);
		}
		#else
		{
			glScissor(left, screenHeight - bottom, right - left, bottom - top);
		}
		#endif
	}



	void CRenderer::setTargetWriteState(bool state)
	{
		#ifdef RNDR_D3D
		{
			if (state)
				D3DDevice->SetRenderState(D3DRS_COLORWRITEENABLE, 0x0000000F);
			else
				D3DDevice->SetRenderState(D3DRS_COLORWRITEENABLE, 0);
		}
		#else
		{
			glColorMask(state, state, state, state);
		}
		#endif
	}



	void CRenderer::setDepthWriteState(bool state)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetRenderState(D3DRS_ZWRITEENABLE, state);
		}
		#else
		{
			glDepthMask(state);
		}
		#endif
	}



	void CRenderer::setDepthTestingFunction(TestingFunction testingFunction)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetRenderState(D3DRS_ZFUNC, testingFunction);
		}
		#else
		{
			glDepthFunc(testingFunction);
		}
		#endif
	}



	void CRenderer::setStencilState(bool state)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetRenderState(D3DRS_STENCILENABLE, state);
		}
		#else
		{
			if (state)
				glEnable(GL_STENCIL_TEST);
			else
				glDisable(GL_STENCIL_TEST);
		}
		#endif
	}



	void CRenderer::setStencilTwoSidedState(bool state)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetRenderState(D3DRS_TWOSIDEDSTENCILMODE, state);
		}
		#else
		{
			if (state)
				glEnable(GL_STENCIL_TEST_TWO_SIDE_EXT);
			else
				glDisable(GL_STENCIL_TEST_TWO_SIDE_EXT);
		}
		#endif
	}



	void CRenderer::setStencilFunction(TestingFunction testingFunction, unsigned int reference, unsigned int mask)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetRenderState(D3DRS_STENCILFUNC, testingFunction);
			D3DDevice->SetRenderState(D3DRS_CCW_STENCILFUNC, testingFunction);

			D3DDevice->SetRenderState(D3DRS_STENCILREF, reference);
			D3DDevice->SetRenderState(D3DRS_STENCILMASK, mask);
		}
		#else
		{
			glActiveStencilFaceEXT(GL_FRONT);
			glStencilFunc(testingFunction, reference, mask);
			glActiveStencilFaceEXT(GL_BACK);
			glStencilFunc(testingFunction, reference, mask);
		}
		#endif
	}



	void CRenderer::setStencilOperation(StencilOperation fail, StencilOperation zfail, StencilOperation pass)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetRenderState(D3DRS_STENCILFAIL, fail);
			D3DDevice->SetRenderState(D3DRS_STENCILZFAIL, zfail);
			D3DDevice->SetRenderState(D3DRS_STENCILPASS, pass);
		}
		#else
		{
			glActiveStencilFaceEXT(GL_FRONT);
			glStencilOp(fail, zfail, pass);
		}
		#endif
	}



	void CRenderer::setStencilOperationBackFace(StencilOperation fail, StencilOperation zfail, StencilOperation pass)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetRenderState(D3DRS_CCW_STENCILFAIL, fail);
			D3DDevice->SetRenderState(D3DRS_CCW_STENCILZFAIL, zfail);
			D3DDevice->SetRenderState(D3DRS_CCW_STENCILPASS, pass);
		}
		#else
		{
			glActiveStencilFaceEXT(GL_BACK);
			glStencilOp(fail, zfail, pass);
		}
		#endif
	}



	void CRenderer::setStencilMask(unsigned int mask)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetRenderState(D3DRS_STENCILWRITEMASK, mask);
		}
		#else
		{
			glStencilMask(mask);
		}
		#endif
	}



	void CRenderer::setBlendingState(bool state)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, state);
		}
		#else
		{
			if (state)
				glEnable(GL_BLEND);
			else
				glDisable(GL_BLEND);
		}
		#endif
	}



	void CRenderer::setBlendingFunction(BlendingFunction srcBlendingFunction, BlendingFunction destBlendingFunction)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetRenderState(D3DRS_SRCBLEND, srcBlendingFunction);
			D3DDevice->SetRenderState(D3DRS_DESTBLEND, destBlendingFunction);
		}
		#else
		{
			glBlendFunc(srcBlendingFunction, destBlendingFunction);
		}
		#endif
	}



	void CRenderer::setRenderTarget(const CRenderTarget *renderTarget)
	{
		if (renderTarget == currentRenderTarget)
			return;

		currentRenderTarget = (CRenderTarget*)renderTarget;

		#ifdef RNDR_D3D
		{
			if (renderTarget == NULL)
				D3DDevice->SetRenderTarget(0, backBufferTargetSurface);
			else
				D3DDevice->SetRenderTarget(0, renderTarget->surface);
		}
		#else
		{
			if (renderTarget == NULL)
			{
				glViewport(0, 0, CApplication::getScreenWidth(), CApplication::getScreenHeight());
				glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
			}
			else
			{
				glViewport(0, 0, renderTarget->width, renderTarget->height);
				glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, offScreenFramebuffer);

				glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, renderTarget->texture, 0);
			}
		}
		#endif
	}



	void CRenderer::setDepthStencilSurface(const CDepthStencilSurface *depthStencilSurface)
	{
		if (depthStencilSurface == currentDepthStencilSurface)
			return;

		currentDepthStencilSurface = (CDepthStencilSurface*)depthStencilSurface;

		#ifdef RNDR_D3D
		{
			if (depthStencilSurface == NULL)
				D3DDevice->SetDepthStencilSurface(backBufferDepthStencilSurface);
			else
				D3DDevice->SetDepthStencilSurface(depthStencilSurface->surface);
		}
		#else
		{
			if (depthStencilSurface == NULL)
			{
				if (currentRenderTarget != NULL)
				{
					glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, offScreenFramebuffer);

					glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, 0, 0);
					glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_TEXTURE_2D, 0, 0);
				}
			}
			else
			{
				glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, offScreenFramebuffer);

				if (depthStencilSurface->format == dssfD24X8 || depthStencilSurface->format == dssfShadowMap)
				{
					glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, depthStencilSurface->texture, 0);
					glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_TEXTURE_2D, 0, 0);
				}
				else
				{
					glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, depthStencilSurface->texture, 0);
					glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_TEXTURE_2D, depthStencilSurface->texture, 0);
				}
			}
		}
		#endif
	}



	void CRenderer::setVertexDeclaration(const CVertexDeclaration &vertexDeclaration)
	{
		if (&vertexDeclaration == currentVertexDeclaration)
			return;

		currentVertexDeclaration = (CVertexDeclaration*)&vertexDeclaration;
		vertexDeclarationChanged = true;

		#ifdef RNDR_D3D
		{
			D3DDevice->SetVertexDeclaration(vertexDeclaration.id);
		}
		#else
		{
			if (vertexDeclaration.positionComponentsNum != 0)
				glEnableClientState(GL_VERTEX_ARRAY);
			else
				glDisableClientState(GL_VERTEX_ARRAY);

			if (vertexDeclaration.colorComponentsNum != 0)
				glEnableClientState(GL_COLOR_ARRAY);
			else
				glDisableClientState(GL_COLOR_ARRAY);

			if (vertexDeclaration.normalComponentsNum != 0)
				glEnableClientState(GL_NORMAL_ARRAY);
			else
				glDisableClientState(GL_NORMAL_ARRAY);

			for (int i = 0; i < 8; i++)
			{
				glClientActiveTexture(GL_TEXTURE0 + i);

				if (vertexDeclaration.texCoordComponentsNum[i] != 0)
					glEnableClientState(GL_TEXTURE_COORD_ARRAY);
				else
					glDisableClientState(GL_TEXTURE_COORD_ARRAY);
			}
		}
		#endif
	}



	void CRenderer::setVertexBuffer(const CVertexBuffer &vertexBuffer)
	{
		if (&vertexBuffer == currentVertexBuffer && !vertexDeclarationChanged)
			return;

		currentVertexBuffer = (CVertexBuffer*)&vertexBuffer;
		vertexDeclarationChanged = false;

		#ifdef RNDR_D3D
		{
			D3DDevice->SetStreamSource(0, vertexBuffer.id, 0, currentVertexDeclaration->size);
		}
		#else
		{
			glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer.id);

			// the code below is here and not in CRenderer::setVertexDeclaration because gl***Pointer must be called after glBindBuffer (gl***Pointer is called for currently being set vertex buffer)

			if (currentVertexDeclaration->positionComponentsNum != 0)
				if (currentVertexDeclaration->positionOffset == 0)
					glVertexPointer(currentVertexDeclaration->positionComponentsNum, GL_FLOAT, currentVertexDeclaration->size, (char*)NULL + 0);
				else
					glVertexPointer(currentVertexDeclaration->positionComponentsNum, GL_FLOAT, currentVertexDeclaration->size, (char*)NULL + currentVertexDeclaration->positionOffset);

			if (currentVertexDeclaration->colorComponentsNum != 0)
				if (currentVertexDeclaration->colorOffset == 0)
					glColorPointer(currentVertexDeclaration->colorComponentsNum, GL_FLOAT, currentVertexDeclaration->size, (char*)NULL + sizeof(float)*currentVertexDeclaration->positionComponentsNum);
				else
					glColorPointer(currentVertexDeclaration->colorComponentsNum, GL_FLOAT, currentVertexDeclaration->size, (char*)NULL + currentVertexDeclaration->colorOffset);

			if (currentVertexDeclaration->normalComponentsNum != 0)
				if (currentVertexDeclaration->normalOffset == 0)
					glNormalPointer(GL_FLOAT, currentVertexDeclaration->size, (char*)NULL + sizeof(float)*(currentVertexDeclaration->positionComponentsNum + currentVertexDeclaration->colorComponentsNum));
				else
					glNormalPointer(GL_FLOAT, currentVertexDeclaration->size, (char*)NULL + currentVertexDeclaration->normalOffset);

			int texCoordsOffset = sizeof(float) * (currentVertexDeclaration->positionComponentsNum + currentVertexDeclaration->colorComponentsNum + currentVertexDeclaration->normalComponentsNum);
			for (int i = 0; i < 8; i++)
			{
				if (currentVertexDeclaration->texCoordComponentsNum[i] != 0)
				{
					glClientActiveTexture(GL_TEXTURE0 + i);

					if (currentVertexDeclaration->texCoordOffset[i] == 0)
						glTexCoordPointer(currentVertexDeclaration->texCoordComponentsNum[i], GL_FLOAT, currentVertexDeclaration->size, (char*)NULL + texCoordsOffset);
					else
						glTexCoordPointer(currentVertexDeclaration->texCoordComponentsNum[i], GL_FLOAT, currentVertexDeclaration->size, (char*)NULL + currentVertexDeclaration->texCoordOffset[i]);	

					texCoordsOffset += sizeof(float) * currentVertexDeclaration->texCoordComponentsNum[i];
				}
			}
		}
		#endif
	}



	void CRenderer::setIndexBuffer(const CIndexBuffer &indexBuffer)
	{
		if (&indexBuffer == currentIndexBuffer)
			return;

		currentIndexBuffer = (CIndexBuffer*)&indexBuffer;

		#ifdef RNDR_D3D
		{
			D3DDevice->SetIndices(indexBuffer.id);
		}
		#else
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.id);
		}
		#endif
	}



	void CRenderer::setShader(const CShader &shader)
	{
		if (shader.getShaderType() == stVertexShader)
		{
			if (&shader == currentVertexShader)
				return;
			else
				currentVertexShader = (CShader*)&shader;
		}
		else
		{
			if (&shader == currentPixelShader)
				return;
			else
				currentPixelShader = (CShader*)&shader;
		}

		#ifdef RNDR_D3D
		{
			cgD3D9BindProgram(shader.cgProgram);
		}
		#else
		{
			cgGLBindProgram(shader.cgProgram);
		}
		#endif
	}



	void CRenderer::setTexture(int sampler, const CTexture &texture)
	{
		if (sampler == 0)
		{
			if (&texture == currentSamplerTexture[0])
				return;
		}
		else if (sampler == 1)
		{
			if (&texture == currentSamplerTexture[1])
				return;
		}
		else if (sampler == 2)
		{
			if (&texture == currentSamplerTexture[2])
				return;
		}
		else if (sampler == 3)
		{
			if (&texture == currentSamplerTexture[3])
				return;
		}

		#ifdef RNDR_D3D
		{
			if (texture.magFiltering != currentSamplerMagFiltering[sampler])
			{
				D3DDevice->SetSamplerState(sampler, D3DSAMP_MAGFILTER, texture.magFiltering);
				currentSamplerMagFiltering[sampler] = texture.magFiltering;
			}
			if (texture.minFiltering != currentSamplerMinFiltering[sampler])
			{
				D3DDevice->SetSamplerState(sampler, D3DSAMP_MINFILTER, texture.minFiltering);
				currentSamplerMinFiltering[sampler] = texture.minFiltering;
			}
			if (texture.mipFiltering != currentSamplerMipFiltering[sampler])
			{
				D3DDevice->SetSamplerState(sampler, D3DSAMP_MIPFILTER, texture.mipFiltering);
				currentSamplerMipFiltering[sampler] = texture.mipFiltering;
			}

			if (texture.addressing != currentSamplerAddressing[sampler])
			{
				D3DDevice->SetSamplerState(sampler, D3DSAMP_ADDRESSU, (D3DTEXTUREADDRESS)texture.addressing);
				D3DDevice->SetSamplerState(sampler, D3DSAMP_ADDRESSV, (D3DTEXTUREADDRESS)texture.addressing);

				currentSamplerAddressing[sampler] = texture.addressing;
			}

			if (texture.borderColor != currentSamplerBorderColor[sampler])
			{
				D3DDevice->SetSamplerState(sampler, D3DSAMP_BORDERCOLOR, D3DCOLOR_XRGB((int)(255.0f*texture.borderColor.x), (int)(255.0f*texture.borderColor.y), (int)(255.0f*texture.borderColor.z)));
				currentSamplerBorderColor[sampler] = texture.borderColor;
			}

			D3DDevice->SetTexture(sampler, texture.id);
		}
		#else
		{
			glActiveTexture(GL_TEXTURE0 + sampler);
			glBindTexture(GL_TEXTURE_2D, texture.id);
		}
		#endif

		if (sampler == 0)
		{
			currentSamplerTexture[0] = (CTexture*)&texture;
		}
		else if (sampler == 1)
		{
			currentSamplerTexture[1] = (CTexture*)&texture;
		}
		else if (sampler == 2)
		{
			currentSamplerTexture[2] = (CTexture*)&texture;
		}
		else if (sampler == 3)
		{
			currentSamplerTexture[3] = (CTexture*)&texture;
		}
	}



	void CRenderer::setTexture(int sampler, const CRenderTarget &renderTarget)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetSamplerState(sampler, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
			D3DDevice->SetSamplerState(sampler, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
			D3DDevice->SetSamplerState(sampler, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
			currentSamplerMagFiltering[sampler] = tfLinear;
			currentSamplerMinFiltering[sampler] = tfLinear;
			currentSamplerMipFiltering[sampler] = tfNone;

			D3DDevice->SetSamplerState(sampler, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
			D3DDevice->SetSamplerState(sampler, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
			currentSamplerAddressing[sampler] = taClamp;

			D3DDevice->SetTexture(sampler, renderTarget.texture);
		}
		#else
		{
			glActiveTexture(GL_TEXTURE0 + sampler);
			glBindTexture(GL_TEXTURE_2D, renderTarget.texture);
		}
		#endif

		currentSamplerTexture[sampler] = NULL;
	}



	void CRenderer::setTexture(int sampler, const CDepthStencilSurface &depthStencilSurface)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->SetSamplerState(sampler, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
			D3DDevice->SetSamplerState(sampler, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
			D3DDevice->SetSamplerState(sampler, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
			currentSamplerMagFiltering[sampler] = tfLinear;
			currentSamplerMinFiltering[sampler] = tfLinear;
			currentSamplerMipFiltering[sampler] = tfNone;

			D3DDevice->SetSamplerState(sampler, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
			D3DDevice->SetSamplerState(sampler, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
			currentSamplerAddressing[sampler] = taClamp;

			if (depthStencilSurface.format == dssfShadowMap)
			{
				D3DDevice->SetSamplerState(sampler, D3DSAMP_ADDRESSU, D3DTADDRESS_BORDER);
				D3DDevice->SetSamplerState(sampler, D3DSAMP_ADDRESSV, D3DTADDRESS_BORDER);

				D3DDevice->SetSamplerState(sampler, D3DSAMP_BORDERCOLOR, D3DCOLOR_XRGB(255, 255, 255));

				currentSamplerAddressing[sampler] = taBorder;
				currentSamplerBorderColor[sampler] = CVector3(1.0f, 1.0f, 1.0f);
			}

			D3DDevice->SetTexture(sampler, depthStencilSurface.texture);
		}
		#else
		{
			glActiveTexture(GL_TEXTURE0 + sampler);
			glBindTexture(GL_TEXTURE_2D, depthStencilSurface.texture);
		}
		#endif

		currentSamplerTexture[sampler] = NULL;
	}



	void CRenderer::drawPrimitives(PrimitiveType primitiveType, int startVertex, int primitivesNum)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->DrawPrimitive((D3DPRIMITIVETYPE)primitiveType, startVertex, primitivesNum);
		}
		#else
		{
			if (primitiveType == ptTriangleList)
				glDrawArrays(ptTriangleList, startVertex, 3*primitivesNum);
			else if (primitiveType == ptTriangleStrip)
				glDrawArrays(ptTriangleStrip, startVertex, primitivesNum + 2);
		}
		#endif
	}



	void CRenderer::drawIndexedPrimitives(PrimitiveType primitiveType, int verticesNum, int primitivesNum)
	{
		#ifdef RNDR_D3D
		{
			D3DDevice->DrawIndexedPrimitive((D3DPRIMITIVETYPE)primitiveType, 0, 0, verticesNum, 0, primitivesNum);
		}
		#else
		{
			if (primitiveType == ptTriangleList)
				glDrawElements(ptTriangleList, 3*primitivesNum, GL_UNSIGNED_SHORT, NULL);
			else if (primitiveType == ptTriangleStrip)
				glDrawElements(ptTriangleStrip, primitivesNum + 2, GL_UNSIGNED_SHORT, NULL);
		}
		#endif
	}
}
