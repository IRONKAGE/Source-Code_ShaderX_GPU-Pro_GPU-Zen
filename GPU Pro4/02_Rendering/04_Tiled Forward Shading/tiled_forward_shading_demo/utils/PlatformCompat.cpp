#include "PlatformCompat.h"

// linux
#if defined(__linux__)
#	include <unistd.h>
#	include <X11/Xlib.h>

/* The best way to query a keystate in X11 seems to be via XQueryKeymap(),
 * which returns 32*8 = 128 bits, one for each key on the keyboard. How these
 * bits map to specific keys is (AFAIK) not easily determined. The values below
 * are therefore not portable.
 */
#	define XQK_LSHIFT_IDX 6
#	define XQK_LSHIFT_MASK 0x4
#	define XQK_RSHIFT_IDX 7
#	define XQK_RSHIFT_MASK 0x40

bool isShiftPressed()
{
  static Display* dpy = XOpenDisplay(0);

  char keyState[32];
  XQueryKeymap( dpy, keyState );

  return (XQK_LSHIFT_MASK == (keyState[XQK_LSHIFT_IDX] & XQK_LSHIFT_MASK))
  	|| (XQK_RSHIFT_MASK == (keyState[XQK_RSHIFT_IDX] & XQK_RSHIFT_MASK))
  ;
}

char* _getcwd( char* buf, size_t size )
{
	return getcwd( buf, size );
}
int _stricmp( const char* a, const char* b )
{
	return strcasecmp( a, b );
}
#endif // __linux__

// windows
#if defined(_WIN32)
#	include "Win32ApiWrapper.h"

bool isShiftPressed()
{
  return (( GetKeyState( VK_LSHIFT   ) < 0 ) || ( GetKeyState( VK_RSHIFT   ) < 0 ));
}

int strcasecmp( const char* a, const char* b )
{
	return _stricmp( a, b );
}
#endif // _WIN32

