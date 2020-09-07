/* $Id: vertex_declaration.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#ifdef RNDR_D3D
	#include <d3d9.h>
#endif

#include "vertex_declaration.h"
#include "renderer.h"
#include "../common/blossom_engine_common.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	int CVertexDeclaration::referenceCounter = 0;



	void CVertexDeclaration::init(int positionComponentsNum, int colorComponentsNum, int normalComponentsNum,
								  int texCoord0ComponentsNum, int texCoord1ComponentsNum, int texCoord2ComponentsNum, int texCoord3ComponentsNum,
								  int texCoord4ComponentsNum, int texCoord5ComponentsNum, int texCoord6ComponentsNum, int texCoord7ComponentsNum,
								  int positionOffset, int colorOffset, int normalOffset,
								  int texCoord0Offset, int texCoord1Offset, int texCoord2Offset, int texCoord3Offset,
								  int texCoord4Offset, int texCoord5Offset, int texCoord6Offset, int texCoord7Offset,
								  int size)
	{
		if (exists)
			return;
		
		this->positionComponentsNum = positionComponentsNum;
		this->colorComponentsNum = colorComponentsNum;
		this->normalComponentsNum = normalComponentsNum;
		this->texCoordComponentsNum[0] = texCoord0ComponentsNum;
		this->texCoordComponentsNum[1] = texCoord1ComponentsNum;
		this->texCoordComponentsNum[2] = texCoord2ComponentsNum;
		this->texCoordComponentsNum[3] = texCoord3ComponentsNum;
		this->texCoordComponentsNum[4] = texCoord4ComponentsNum;
		this->texCoordComponentsNum[5] = texCoord5ComponentsNum;
		this->texCoordComponentsNum[6] = texCoord6ComponentsNum;
		this->texCoordComponentsNum[7] = texCoord7ComponentsNum;

		this->positionOffset = positionOffset;
		this->colorOffset = colorOffset;
		this->normalOffset = normalOffset;
		this->texCoordOffset[0] = texCoord0Offset;
		this->texCoordOffset[1] = texCoord1Offset;
		this->texCoordOffset[2] = texCoord2Offset;
		this->texCoordOffset[3] = texCoord3Offset;
		this->texCoordOffset[4] = texCoord4Offset;
		this->texCoordOffset[5] = texCoord5Offset;
		this->texCoordOffset[6] = texCoord6Offset;
		this->texCoordOffset[7] = texCoord7Offset;

		int texCoordsSize = 0;
		for (int i = 0; i < 8; i++)
			texCoordsSize += this->texCoordComponentsNum[i];

		if (size == 0)
			this->size = sizeof(float) * (positionComponentsNum + colorComponentsNum + normalComponentsNum + texCoordsSize);
		else
			this->size = size;
		
		#ifdef RNDR_D3D
		{
			int vertexElementsNum = 0;

			if (positionComponentsNum != 0)
				vertexElementsNum++;
			if (colorComponentsNum != 0)
				vertexElementsNum++;
			if (normalComponentsNum != 0)
				vertexElementsNum++;
			for (int i = 0; i < 8; i++)
			{
				if (texCoordComponentsNum[i] != 0)
					vertexElementsNum++;
			}
			vertexElementsNum++; // one more for D3DDECL_END

			D3DVERTEXELEMENT9 *vertexElements = new D3DVERTEXELEMENT9[vertexElementsNum];
			{
				int totalOffset = 0;
				int currentVertexElementIndex = 0;

				if (positionComponentsNum != 0)
				{
					vertexElements[currentVertexElementIndex].Stream = 0;
					if (positionOffset == 0)
						vertexElements[currentVertexElementIndex].Offset = totalOffset;
					else
						vertexElements[currentVertexElementIndex].Offset = positionOffset;
					vertexElements[currentVertexElementIndex].Type = positionComponentsNum - 1; // see D3DDECLTYPE definition
					vertexElements[currentVertexElementIndex].Method = D3DDECLMETHOD_DEFAULT;
					vertexElements[currentVertexElementIndex].Usage = D3DDECLUSAGE_POSITION;
					vertexElements[currentVertexElementIndex].UsageIndex = 0;

					currentVertexElementIndex++;
					totalOffset += sizeof(float) * positionComponentsNum;
				}

				if (colorComponentsNum != 0)
				{
					vertexElements[currentVertexElementIndex].Stream = 0;
					if (colorOffset == 0)
						vertexElements[currentVertexElementIndex].Offset = totalOffset;
					else
						vertexElements[currentVertexElementIndex].Offset = colorOffset;
					vertexElements[currentVertexElementIndex].Type = colorComponentsNum - 1; // see D3DDECLTYPE definition
					vertexElements[currentVertexElementIndex].Method = D3DDECLMETHOD_DEFAULT;
					vertexElements[currentVertexElementIndex].Usage = D3DDECLUSAGE_COLOR;
					vertexElements[currentVertexElementIndex].UsageIndex = 0;

					currentVertexElementIndex++;
					totalOffset += sizeof(float) * colorComponentsNum;
				}

				if (normalComponentsNum != 0)
				{
					vertexElements[currentVertexElementIndex].Stream = 0;
					if (normalOffset == 0)
						vertexElements[currentVertexElementIndex].Offset = totalOffset;
					else
						vertexElements[currentVertexElementIndex].Offset = normalOffset;
					vertexElements[currentVertexElementIndex].Type = 2; // see D3DDECLTYPE definition
					vertexElements[currentVertexElementIndex].Method = D3DDECLMETHOD_DEFAULT;
					vertexElements[currentVertexElementIndex].Usage = D3DDECLUSAGE_NORMAL;
					vertexElements[currentVertexElementIndex].UsageIndex = 0;

					currentVertexElementIndex++;
					totalOffset += sizeof(float) * normalComponentsNum;
				}

				for (int i = 0; i < 8; i++)
				{
					if (texCoordComponentsNum[i] != 0)
					{
						vertexElements[currentVertexElementIndex].Stream = 0;
						if (texCoordOffset[i] == 0)
							vertexElements[currentVertexElementIndex].Offset = totalOffset;
						else
							vertexElements[currentVertexElementIndex].Offset = texCoordOffset[i];
						vertexElements[currentVertexElementIndex].Type = texCoordComponentsNum[i] - 1; // see D3DDECLTYPE definition
						vertexElements[currentVertexElementIndex].Method = D3DDECLMETHOD_DEFAULT;
						vertexElements[currentVertexElementIndex].Usage = D3DDECLUSAGE_TEXCOORD;
						vertexElements[currentVertexElementIndex].UsageIndex = i;

						currentVertexElementIndex++;
						totalOffset += sizeof(float) * texCoordComponentsNum[i];
					}
				}

				// D3DDECL_END
				{
					vertexElements[currentVertexElementIndex].Stream = 0xFF;
					vertexElements[currentVertexElementIndex].Offset = 0;
					vertexElements[currentVertexElementIndex].Type = D3DDECLTYPE_UNUSED;
					vertexElements[currentVertexElementIndex].Method = 0;
					vertexElements[currentVertexElementIndex].Usage = 0;
					vertexElements[currentVertexElementIndex].UsageIndex = 0;
				}

				CRenderer::D3DDevice->CreateVertexDeclaration(vertexElements, &id);
			}
			delete[] vertexElements;
		}
		#else
		{
			// nothing to be done
		}
		#endif

		exists = true;
		referenceCounter++;
		CLogger::addText("\tOK: vertex declaration created (reference counter = %d)\n", referenceCounter);
	}



	void CVertexDeclaration::free()
	{
		if (!exists)
			return;

		#ifdef RNDR_D3D
		{
			id->Release();
		}
		#else
		{
			// nothing to be done
		}
		#endif

		exists = false;
		referenceCounter--;
		CLogger::addText("\tOK: vertex declaration freed (reference counter = %d)\n", referenceCounter);
	}



	int CVertexDeclaration::getSize()
	{
		return size;
	}
}
