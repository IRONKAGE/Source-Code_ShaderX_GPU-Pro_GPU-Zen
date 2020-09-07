/* $Id: index_buffer.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#ifdef RNDR_D3D
	#include <d3d9.h>
#else
	#include <GL/glew.h>
#endif

#include "index_buffer.h"
#include "renderer.h"
#include "../common/blossom_engine_common.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	int CIndexBuffer::referenceCounter = 0;



	void CIndexBuffer::init(int size)
	{
		if (exists)
			return;

		#ifdef RNDR_D3D
		{
			if (FAILED(CRenderer::D3DDevice->CreateIndexBuffer(size, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED, &id, NULL)))
			{
				CLogger::addText("\tERROR: couldn't create index buffer\n");
				return;
			}
		}
		#else
		{
			glGenBuffers(1, &id);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, NULL, GL_STATIC_DRAW);

			if (!glIsBuffer(id))
			{
				CLogger::addText("\tERROR: couldn't create index buffer\n");
				exit(1);
			}

			CRenderer::currentIndexBuffer = this;
		}
		#endif

		this->size = size;

		exists = true;
		referenceCounter++;
		CLogger::addText("\tOK: index buffer created (reference counter = %d)\n", referenceCounter);
	}



	void CIndexBuffer::free()
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
		CLogger::addText("\tOK: index buffer freed (reference counter = %d)\n", referenceCounter);
	}



	int CIndexBuffer::getSize()
	{
		return size;
	}



	void CIndexBuffer::map(void *&data)
	{
		#ifdef RNDR_D3D
		{
			id->Lock(0, 0, (void**)&data, 0);
		}
		#else
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id);
			data = glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

			CRenderer::currentIndexBuffer = this;
		}
		#endif
	}



	void CIndexBuffer::unmap()
	{
		#ifdef RNDR_D3D
		{
			id->Unlock();
		}
		#else
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id);
			glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);

			CRenderer::currentIndexBuffer = this;
		}
		#endif
	}
}
