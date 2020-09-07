/* $Id: logger.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#include <stdarg.h>

#include "logger.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	FILE *CLogger::file;



	void CLogger::open()
	{
		#ifdef RNDR_D3D
		{
			file = fopen("log_BlossomEngine_D3D9.txt", "w");
		}
		#else
		{
			file = fopen("log_BlossomEngine_OGL.txt", "w");
		}
		#endif
	}



	void CLogger::close()
	{
		fclose(file);
	}



	void CLogger::addText(const char *text, ...)
	{
		char buffer[1024];
		va_list list;

		va_start(list, text);
			vsnprintf(buffer, sizeof(buffer), text, list);
		va_end(list);

		fprintf(file, buffer);
		fflush(file);
	}
}
