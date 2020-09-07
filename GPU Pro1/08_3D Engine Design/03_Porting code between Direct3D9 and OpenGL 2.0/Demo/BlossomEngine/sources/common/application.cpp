/* $Id: application.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#include <SDL/SDL.h>

#ifdef RNDR_D3D
	#include <d3d9.h>
#endif

#include "application.h"
#include "logger.h"
#include "../renderer/blossom_engine_renderer.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	int CApplication::lastFrameTime, CApplication::timeBeforeFrame, CApplication::timeAfterFrame;

	int CApplication::screenWidth, CApplication::screenHeight, CApplication::screenBitsPerPixel;
	bool CApplication::keys[512];
	int CApplication::mouseX, CApplication::mouseY, CApplication::mouseRelX, CApplication::mouseRelY;
	bool CApplication::mouseLeftButtonPressed, CApplication::mouseMiddleButtonPressed, CApplication::mouseRightButtonPressed;

    void (*CApplication::keyDownFunction)(int key);
    void (*CApplication::keyUpFunction)(int key);
	void (*CApplication::mouseMotionFunction)();
	void (*CApplication::mouseLeftButtonDownFunction)();
	void (*CApplication::mouseMiddleButtonDownFunction)();
	void (*CApplication::mouseRightButtonDownFunction)();
	void (*CApplication::mouseLeftButtonUpFunction)();
	void (*CApplication::mouseMiddleButtonUpFunction)();
	void (*CApplication::mouseRightButtonUpFunction)();

	std::vector<CApplication::TimerInfo> CApplication::timersInfo;



	void CApplication::init(int width, int height, int bpp, bool fullScreen, bool vsync)
	{
		const SDL_VideoInfo* info = NULL;

		CLogger::open();
		CLogger::addText("OK: Application start\n");

		if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
		{
			CLogger::addText("Error: SDL initialization failed: %s\n", SDL_GetError());
			free();
			exit(1);
		}
		info = SDL_GetVideoInfo();

		if (width == 0 && height == 0)
		{
			screenWidth = info->current_w;
			screenHeight = info->current_h;
		}
		else
		{
			screenWidth = width;
			screenHeight = height;
		}
		screenBitsPerPixel = info->vfmt->BitsPerPixel;

		int flags = 0;
		if (fullScreen)
			flags |= SDL_FULLSCREEN;

		SDL_putenv("SDL_VIDEO_CENTERED=1");

		#ifdef RNDR_D3D
		{
			if (SDL_SetVideoMode(width, height, bpp, flags) == 0)
			{
				CLogger::addText("Error: Video mode set failed: %s\n", SDL_GetError());
				free();
				exit(1);
			}

			CRenderer::D3DObject = Direct3DCreate9(D3D_SDK_VERSION);

			D3DDISPLAYMODE displayMode;
			CRenderer::D3DObject->GetAdapterDisplayMode(D3DADAPTER_DEFAULT, &displayMode);

			ZeroMemory(&CRenderer::presentParameters, sizeof(CRenderer::presentParameters));
			CRenderer::presentParameters.BackBufferCount = 1;
			CRenderer::presentParameters.BackBufferFormat = displayMode.Format;
			CRenderer::presentParameters.BackBufferWidth = screenWidth;
			CRenderer::presentParameters.BackBufferHeight = screenHeight;
			if (fullScreen)
				CRenderer::presentParameters.FullScreen_RefreshRateInHz = displayMode.RefreshRate;
			else
				CRenderer::presentParameters.FullScreen_RefreshRateInHz = 0;
			CRenderer::presentParameters.EnableAutoDepthStencil = true;
			CRenderer::presentParameters.AutoDepthStencilFormat = D3DFMT_D24S8;
			CRenderer::presentParameters.hDeviceWindow = GetActiveWindow();
			if (vsync)
				CRenderer::presentParameters.PresentationInterval = D3DPRESENT_INTERVAL_DEFAULT;
			else
				CRenderer::presentParameters.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
			CRenderer::presentParameters.SwapEffect = D3DSWAPEFFECT_DISCARD;
			CRenderer::presentParameters.Windowed = !fullScreen;

			CRenderer::D3DObject->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, GetActiveWindow(), D3DCREATE_HARDWARE_VERTEXPROCESSING, &CRenderer::presentParameters, &CRenderer::D3DDevice);
		}
		#else
		{
			SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
			SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
			SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
			SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);

			SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
			SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

			SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, true);
			SDL_GL_SetAttribute(SDL_GL_SWAP_CONTROL, vsync ? 1 : 0);

			flags |= SDL_OPENGL;

			if (SDL_SetVideoMode(width, height, bpp, flags) == 0)
			{
				CLogger::addText("Error: Video mode set failed: %s\n", SDL_GetError());
				free();
				exit(1);
			}
		}
		#endif

		SDL_WM_GrabInput(SDL_GRAB_ON);
		SDL_ShowCursor(false);
		SDL_WM_SetCaption("BlossomEngineApplication", "BlossomEngineApplication");

        keyDownFunction = NULL;
        keyUpFunction = NULL;
		mouseMotionFunction = NULL;
		mouseLeftButtonDownFunction = NULL;
		mouseMiddleButtonDownFunction = NULL;
		mouseRightButtonDownFunction = NULL;
		mouseLeftButtonUpFunction = NULL;
		mouseMiddleButtonUpFunction = NULL;
		mouseRightButtonUpFunction = NULL;
	}



	void CApplication::free()
	{
	    keyDownFunction = NULL;
	    keyUpFunction = NULL;
		mouseMotionFunction = NULL;
		mouseLeftButtonDownFunction = NULL;
		mouseMiddleButtonDownFunction = NULL;
		mouseRightButtonDownFunction = NULL;
		mouseLeftButtonUpFunction = NULL;
		mouseMiddleButtonUpFunction = NULL;
		mouseRightButtonUpFunction = NULL;

		for (unsigned int i = 0; i < timersInfo.size(); i++)
			SDL_RemoveTimer(timersInfo[i].id_SDL);
		timersInfo.clear();

		SDL_ShowCursor(true);
		SDL_Quit();

		CLogger::addText("OK: Application quit\n");
		CLogger::close();
	}



	void CApplication::processEvents()
	{
		SDL_Event event;

		while(SDL_PollEvent(&event))
		{
			switch(event.type)
			{
			case SDL_KEYDOWN:
                if (keyDownFunction != NULL)
                    keyDownFunction((int)event.key.keysym.sym);
				keys[event.key.keysym.sym] = true;
				break;

			case SDL_KEYUP:
                if (keyUpFunction != NULL)
                    keyUpFunction((int)event.key.keysym.sym);
				keys[event.key.keysym.sym] = false;
				break;

			case SDL_MOUSEMOTION:
				if (mouseMotionFunction != NULL)
					mouseMotionFunction();
				mouseX = event.motion.x;
				mouseY = event.motion.y;
				mouseRelX = event.motion.xrel;
				mouseRelY = event.motion.yrel;
				break;

			case SDL_MOUSEBUTTONDOWN:
				if (event.button.button == SDL_BUTTON_LEFT)
				{
					if (mouseLeftButtonDownFunction != NULL)
						mouseLeftButtonDownFunction();
					mouseLeftButtonPressed = true;
				}
				else if (event.button.button == SDL_BUTTON_MIDDLE)
				{
					if (mouseMiddleButtonDownFunction != NULL)
						mouseMiddleButtonDownFunction();
					mouseMiddleButtonPressed = true;
				}
				else if (event.button.button == SDL_BUTTON_RIGHT)
				{
					if (mouseRightButtonDownFunction != NULL)
						mouseRightButtonDownFunction();
					mouseRightButtonPressed = true;
				}
				break;

			case SDL_MOUSEBUTTONUP:
				if (event.button.button == SDL_BUTTON_LEFT)
				{
					if (mouseLeftButtonUpFunction != NULL)
						mouseLeftButtonUpFunction();
					mouseLeftButtonPressed = false;
				}
				else if (event.button.button == SDL_BUTTON_MIDDLE)
				{
					if (mouseMiddleButtonUpFunction != NULL)
						mouseMiddleButtonUpFunction();
					mouseMiddleButtonPressed = false;
				}
				else if (event.button.button == SDL_BUTTON_RIGHT)
				{
					if (mouseRightButtonUpFunction != NULL)
						mouseRightButtonUpFunction();
					mouseRightButtonPressed = false;
				}
				break;

			case SDL_USEREVENT:
				for (unsigned int i = 0; i < timersInfo.size(); i++)
				{
					if (event.user.code == timersInfo[i].id)
						timersInfo[i].eventFunction();
				}
				break;
			}
		}
	}



	void CApplication::setKeyDownFunction(void (*function)(int key))
	{
        keyDownFunction = function;
	}



	void CApplication::setKeyUpFunction(void (*function)(int key))
	{
        keyUpFunction = function;
	}



	void CApplication::setMousePosition(int x, int y)
	{
		SDL_WarpMouse(x, y);
	}



	void CApplication::setMouseMotionFunction(void (*function)())
	{
		mouseMotionFunction = function;
	}



	void CApplication::setMouseLeftButtonDownFunction(void (*function)())
	{
		mouseLeftButtonDownFunction = function;
	}



	void CApplication::setMouseMiddleButtonDownFunction(void (*function)())
	{
		mouseMiddleButtonDownFunction = function;
	}



	void CApplication::setMouseRightButtonDownFunction(void (*function)())
	{
		mouseRightButtonDownFunction = function;
	}



	void CApplication::setMouseLeftButtonUpFunction(void (*function)())
	{
		mouseLeftButtonUpFunction = function;
	}



	void CApplication::setMouseMiddleButtonUpFunction(void (*function)())
	{
		mouseMiddleButtonUpFunction = function;
	}



	void CApplication::setMouseRightButtonUpFunction(void (*function)())
	{
		mouseRightButtonUpFunction = function;
	}



	void CApplication::setWindowText(const char *text)
	{
		SDL_WM_SetCaption(text, text);
	}



	void CApplication::showCursor(bool toggle)
	{
		SDL_ShowCursor(toggle);
	}



	int CApplication::getTickCount()
	{
		return SDL_GetTicks();
	}



	void CApplication::beginLoop()
	{
		lastFrameTime = timeAfterFrame - timeBeforeFrame;
		timeBeforeFrame = SDL_GetTicks();

		// yeah, a lot of code here - but need to handle lost device in D3D
		#ifdef RNDR_D3D
		{
			HRESULT result = CRenderer::D3DDevice->TestCooperativeLevel();

			if (result == D3DERR_DEVICENOTRESET)
			{
				CLogger::addText("OK: Device lost\n");

				CRenderer::backBufferTargetSurface->Release();
				CRenderer::backBufferDepthStencilSurface->Release();

				for (unsigned int i = 0; i < CRenderer::renderTargets.size(); i++)
					CRenderer::renderTargets[i]->free();
				for (unsigned int i = 0; i < CRenderer::depthStencilSurfaces.size(); i++)
					CRenderer::depthStencilSurfaces[i]->free();

				CRenderer::D3DDevice->Reset(&CRenderer::presentParameters);
				CLogger::addText("OK: Device reset\n");

				CRenderer::D3DDevice->GetRenderTarget(0, &CRenderer::backBufferTargetSurface);
				CRenderer::D3DDevice->GetDepthStencilSurface(&CRenderer::backBufferDepthStencilSurface);

				CRenderer::D3DDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
				CRenderer::D3DDevice->SetRenderState(D3DRS_ZFUNC, D3DCMP_LESSEQUAL);

				for (unsigned int i = 0; i < CRenderer::renderTargets.size(); i++)
					CRenderer::renderTargets[i]->init(CRenderer::renderTargets[i]->getFormat(), CRenderer::renderTargets[i]->getWidth(), CRenderer::renderTargets[i]->getHeight(), false);
				for (unsigned int i = 0; i < CRenderer::depthStencilSurfaces.size(); i++)
					CRenderer::depthStencilSurfaces[i]->init(CRenderer::depthStencilSurfaces[i]->getFormat(), CRenderer::depthStencilSurfaces[i]->getWidth(), CRenderer::depthStencilSurfaces[i]->getHeight(), false);

				CRenderer::currentCullMode = cmCW;
				CRenderer::currentRenderTarget = NULL;
				CRenderer::currentDepthStencilSurface = NULL;
				CRenderer::currentVertexDeclaration = NULL;
				CRenderer::currentVertexBuffer = NULL;
				CRenderer::currentIndexBuffer = NULL;
				CRenderer::currentVertexShader = NULL;
				CRenderer::currentPixelShader = NULL;
				for (int i = 0; i < 4; i++)
				{
					CRenderer::currentSamplerTexture[i] = NULL;
					CRenderer::currentSamplerMagFiltering[i] = (TextureFiltering)-1;
					CRenderer::currentSamplerMinFiltering[i] = (TextureFiltering)-1;
					CRenderer::currentSamplerMipFiltering[i] = (TextureFiltering)-1;
					CRenderer::currentSamplerAddressing[i] = (TextureAddressing)-1;
					CRenderer::currentSamplerBorderColor[i] = CVector3(-1.0f, -1.0f, -1.0f);
				}

				CLogger::addText("OK: Resources recreated\n");
			}
			else if (result == D3DERR_DEVICELOST)
			{
				SDL_Delay(1000);
				return;
			}

			CRenderer::D3DDevice->BeginScene();
		}
		#else
		{
			// nothing to be done
		}
		#endif
	}



	void CApplication::endLoop()
	{
		#ifdef RNDR_D3D
		{
			CRenderer::D3DDevice->EndScene();
			CRenderer::D3DDevice->Present(0, 0, 0, 0);
		}
		#else
		{
			SDL_GL_SwapBuffers();
		}
		#endif

		timeAfterFrame = SDL_GetTicks();
	}



	int CApplication::getLastFrameTime()
	{
		return lastFrameTime;
	}



	int CApplication::getScreenWidth()
	{
		return screenWidth;
	}



	int CApplication::getScreenHeight()
	{
		return screenHeight;
	}



	int CApplication::getScreenBitsPerPixel()
	{
		return screenBitsPerPixel;
	}



	bool CApplication::isKeyPressed(int key)
	{
		return keys[key];
	}



	int CApplication::getMouseX()
	{
		return mouseX;
	}



	int CApplication::getMouseY()
	{
		return mouseY;
	}



	int CApplication::getMouseRelX()
	{
		return mouseRelX;
	}



	int CApplication::getMouseRelY()
	{
		return mouseRelY;
	}



	bool CApplication::isMouseLeftButtonPressed()
	{
		return mouseLeftButtonPressed;
	}



	bool CApplication::isMouseMiddleButtonPressed()
	{
		return mouseMiddleButtonPressed;
	}



	bool CApplication::isMouseRightButtonPressed()
	{
		return mouseRightButtonPressed;
	}



	void CApplication::addTimer(const int &id, int interval, void(*eventFunction)())
	{
		TimerInfo timerInfo;

		timerInfo.id = id;
		timerInfo.id_SDL = SDL_AddTimer(interval, timerFunc, (void*)&id);
		timerInfo.interval = interval;
		timerInfo.eventFunction = eventFunction;

		timersInfo.push_back(timerInfo);
	}



	void CApplication::deleteTimer(int id)
	{
		for (unsigned int i = 0; i < timersInfo.size(); i++)
		{
			if (timersInfo[i].id == id)
			{
				SDL_RemoveTimer(timersInfo[i].id_SDL);
				timersInfo.erase(timersInfo.begin() + i);
				break;
			}
		}
	}



	Uint32 CApplication::timerFunc(Uint32 interval, void *param)
	{
		SDL_Event event;

		event.type = SDL_USEREVENT;
		event.user.code = *((int*)(param));
		event.user.data1 = 0;
		event.user.data2 = 0;

		SDL_PushEvent(&event);

		return interval;
	}
}
