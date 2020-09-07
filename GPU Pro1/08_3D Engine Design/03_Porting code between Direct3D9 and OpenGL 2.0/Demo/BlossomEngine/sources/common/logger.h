/* $Id: logger.h 126 2009-08-22 17:08:39Z maxest $ */

#ifndef _BLOSSOM_ENGINE_LOGGER_
#define _BLOSSOM_ENGINE_LOGGER_

#include <stdio.h>

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CLogger
	{
	private:
		static FILE *file;

	public:
		static void open();
		static void close();
		static void addText(const char *text, ...);
	};
}

// ----------------------------------------------------------------------------

#endif
