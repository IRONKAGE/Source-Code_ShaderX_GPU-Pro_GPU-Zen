/* $Id: vertex_buffer.h 148 2009-08-24 17:28:38Z maxest $ */

#ifndef _BLOSSOM_ENGINE_VERTEX_BUFFER_
#define _BLOSSOM_ENGINE_VERTEX_BUFFER_

#ifdef RNDR_D3D
	#include <d3d9.h>
#endif

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CRenderer;

	// ----------------------------------------------------------------------------

	class CVertexBuffer
	{
		friend class CRenderer;

	public:
		static int referenceCounter;
		bool exists;

		#ifdef RNDR_D3D
			IDirect3DVertexBuffer9 *id;
		#else
			unsigned int id;
		#endif

		int size;

	public:
		CVertexBuffer() { exists = false; }

		void init(int size);
		void free();

		int getSize();

		void map(void *&data);
		void unmap();
	};
}

#endif
