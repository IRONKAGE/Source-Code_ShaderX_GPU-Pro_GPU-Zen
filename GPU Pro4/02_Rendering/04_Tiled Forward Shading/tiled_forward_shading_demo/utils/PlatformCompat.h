#ifndef PLATFORMCOMPAT_H_3E639E41_946D_4BC5_A3B7_A7C4F5821925
#define PLATFORMCOMPAT_H_3E639E41_946D_4BC5_A3B7_A7C4F5821925

// common stuff
bool isShiftPressed();

// platform specific fixes
#if defined(__linux__)
#	include <cstring>
#	include <unistd.h>

char* _getcwd( char* buf, size_t size );

int _stricmp( const char* a, const char* b );

#	define _snprintf snprintf //HACK

#	define WARN_STRINGIZE0_(w) #w
#	define WARN_STRINGIZE1_(w,s) WARN_STRINGIZE0_(GCC diagnostic s w)
#	define WARN_STRINGIZE2_(w,s) WARN_STRINGIZE1_(#w,s)

#	if __GNUC__ >= 4 && __GNUC_MINOR__ >= 6
#		define BEGIN_WARNING_CLOBBER_GCC(warn) \
			_Pragma( "GCC diagnostic push" ) \
			_Pragma( WARN_STRINGIZE2_(warn, ignored) ) \
			/*ENDM*/
#		define END_WARNING_CLOBBER_GCC(warn) \
			_Pragma( "GCC diagnostic pop" ) \
			/*ENDM*/
#	else
#		define BEGIN_WARNING_CLOBBER_GCC(warn) \
			_Pragma( WARN_STRINGIZE2_(warn, ignored) ) \
			/*ENDM*/
#		define END_WARNING_CLOBBER_GCC(warn) \
			_Pragma( WARN_STRINGIZE2_(warn, warning) ) \
			/*ENDM*/
#	endif

#define BEGIN_WARNING_CLOBBER_MSVC 
#define END_WARNING_CLOBBER_MSVC 
#define DISABLE_SPECIFIC_WARNING_MSCV(_warn_num_) 


#elif defined(_WIN32)
#	include <direct.h>
#	include <cstring>

int strcasecmp( const char* a, const char* b );

#define BEGIN_WARNING_CLOBBER_MSVC __pragma(warning( push,3 ))
#define END_WARNING_CLOBBER_MSVC __pragma(warning( pop ))
#define DISABLE_SPECIFIC_WARNING_MSCV(_warn_num_) __pragma(warning( disable: _warn_num_))

#	define BEGIN_WARNING_CLOBBER_GCC(w)
#	define END_WARNING_CLOBBER_GCC(w)

#endif // ! _WIN32 || __linux

#endif // PLATFORMCOMPAT_H_3E639E41_946D_4BC5_A3B7_A7C4F5821925
