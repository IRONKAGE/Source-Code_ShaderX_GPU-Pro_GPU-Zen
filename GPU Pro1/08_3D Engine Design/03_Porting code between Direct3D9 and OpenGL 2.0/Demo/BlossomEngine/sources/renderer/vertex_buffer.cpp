/* $Id: vertex_buffer.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#ifdef RNDR_D3D
	#include <d3d9.h>
#else
	#include <GL/glew.h>
#endif

#include "vertex_buffer.h"
#include "renderer.h"
#include "../common/blossom_engine_common.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	int CVertexBuffer::referenceCounter = 0;



	void CVertexBuffer::init(int size)
	{
		if (exists)
			return;

		#ifdef RNDR_D3D
		{
			if (FAILED(CRenderer::D3DDevice->CreateVertexBuffer(size, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED, &id, NULL)))
			{
				CLogger::addText("\tERROR: couldn't create vertex buffer\n");
				return;
			}
		}
		#else
		{
			glGenBuffers(1, &id);
			glBindBuffer(GL_ARRAY_BUFFER, id);
			glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_STATIC_DRAW);

			if (!glIsBuffer(id))
			{
				CLogger::addText("\tERROR: couldn't create vertex buffer\n");
				exit(1);
			}

			CRenderer::currentVertexBuffer = this;
		}
		#endif

		this->size = size;

		exists = true;
		referenceCounter++;
		CLogger::addText("\tOK: vertex buffer created (reference counter = %d)\n", referenceCounter);
	}



	void CVertexBuffer::free()
	{
		if (!exists)
			return;

		#ifdef RNDR_D3D
		{
			id->Release();
		}
		#else
		{
			glDeleteBuffers(1, &id);
		}
		#endif

		exists = false;
		referenceCounter--;
		CLogger::addText("\tOK: vertex buffer freed (reference counter = %d)\n", referenceCounter);
	}



	int CVertexBuffer::getSize()
	{
		return size;
	}



	void CVertexBuffer::map(void *&data)
	{
		#ifdef RNDR_D3D
		{
			id->Lock(0, 0, (void**)&data, 0);
		}
		#else
		{
			glBindBuffer(GL_ARRAY_BUFFER, id);
			data = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

			CRenderer::currentVertexBuffer = this;
		}
		#endif
	}



	void CVertexBuffer::unmap()
	{
		#ifdef RNDR_D3D
		{
			id->Unlock();
		}
		#else
		{
			glBindBuffer(GL_ARRAY_BUFFER, id);
			glUnmapBuffer(GL_ARRAY_BUFFER);

			CRenderer::currentVertexBuffer = this;
		}
		#endif
	}
}
