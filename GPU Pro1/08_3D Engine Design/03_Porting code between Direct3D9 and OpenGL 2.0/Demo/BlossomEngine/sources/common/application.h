/* $Id: application.h 135 2009-08-22 23:23:33Z maxest $ */

#ifndef _BLOSSOM_ENGINE_APPLICATION_
#define _BLOSSOM_ENGINE_APPLICATION_

#include <SDL/SDL.h>

#include <vector>

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CApplication
	{
		// ----------------------------------------------------------------------------

		struct TimerInfo
		{
			int id;
			SDL_TimerID id_SDL;
			int interval;
			void (*eventFunction)();
		};

		// ----------------------------------------------------------------------------

	private:
		static int lastFrameTime, timeBeforeFrame, timeAfterFrame;

		static int screenWidth, screenHeight, screenBitsPerPixel;
		static bool keys[512];
		static int mouseX, mouseY, mouseRelX, mouseRelY;
		static bool mouseLeftButtonPressed, mouseMiddleButtonPressed, mouseRightButtonPressed;

        static void (*keyDownFunction)(int key);
        static void (*keyUpFunction)(int key);
		static void (*mouseMotionFunction)();
		static void (*mouseLeftButtonDownFunction)();
		static void (*mouseMiddleButtonDownFunction)();
		static void (*mouseRightButtonDownFunction)();
		static void (*mouseLeftButtonUpFunction)();
		static void (*mouseMiddleButtonUpFunction)();
		static void (*mouseRightButtonUpFunction)();

		static std::vector<TimerInfo> timersInfo;
		static Uint32 timerFunc(Uint32 interval, void *param);

	public:
		static void init(int width, int height, int bpp, bool fullScreen, bool vsync);
		static void free();

		static void processEvents();

		static void setKeyDownFunction(void (*function)(int key));
		static void setKeyUpFunction(void (*function)(int key));
		static void setMousePosition(int x, int y);
		static void setMouseMotionFunction(void (*function)());
		static void setMouseLeftButtonDownFunction(void (*function)());
		static void setMouseMiddleButtonDownFunction(void (*function)());
		static void setMouseRightButtonDownFunction(void (*function)());
		static void setMouseLeftButtonUpFunction(void (*function)());
		static void setMouseMiddleButtonUpFunction(void (*function)());
		static void setMouseRightButtonUpFunction(void (*function)());

		static void setWindowText(const char *text);
		static void showCursor(bool toggle);

		static int getTickCount();
		static void beginLoop();
		static void endLoop();
		static int getLastFrameTime();

		static int getScreenWidth();
		static int getScreenHeight();
		static int getScreenBitsPerPixel();

		static bool isKeyPressed(int key);

		static int getMouseX();
		static int getMouseY();
		static int getMouseRelX();
		static int getMouseRelY();

		static bool isMouseLeftButtonPressed();
		static bool isMouseMiddleButtonPressed();
		static bool isMouseRightButtonPressed();

		static void addTimer(const int &id, int interval, void(*eventFunction)());
		static void deleteTimer(int id);
	};
}

// ----------------------------------------------------------------------------

#endif
