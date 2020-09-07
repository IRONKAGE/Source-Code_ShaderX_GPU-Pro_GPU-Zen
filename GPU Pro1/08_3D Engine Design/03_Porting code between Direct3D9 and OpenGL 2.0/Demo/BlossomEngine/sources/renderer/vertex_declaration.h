/* $Id: vertex_declaration.h 148 2009-08-24 17:28:38Z maxest $ */

#ifndef _BLOSSOM_ENGINE_VERTEX_DECLARATION_
#define _BLOSSOM_ENGINE_VERTEX_DECLARATION_

#ifdef RNDR_D3D
	#include <d3d9.h>
#endif

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CRenderer;

	// ----------------------------------------------------------------------------

	class CVertexDeclaration
	{
		friend class CRenderer;

	private:
		static int referenceCounter;
		bool exists;

		#ifdef RNDR_D3D
			IDirect3DVertexDeclaration9 *id;
		#endif

		int size;	
		int positionComponentsNum, colorComponentsNum, normalComponentsNum, texCoordComponentsNum[8];
		int positionOffset, colorOffset, normalOffset, texCoordOffset[8];

	public:
		CVertexDeclaration() { exists = false; }

		// for normalComponentsNum specify either 0 (no normal data) or 3
		void init(int positionComponentsNum = 4, int colorComponentsNum = 0, int normalComponentsNum = 0,
				  int texCoord0ComponentsNum = 2, int texCoord1ComponentsNum = 0, int texCoord2ComponentsNum = 0, int texCoord3ComponentsNum = 0,
				  int texCoord4ComponentsNum = 0, int texCoord5ComponentsNum = 0, int texCoord6ComponentsNum = 0, int texCoord7ComponentsNum = 0,
				  // if the offset equals to 0, it's computed automatically
				  int positionOffset = 0, int colorOffset = 0, int normalOffset = 0,
				  int texCoord0Offset = 0, int texCoord1Offset = 0, int texCoord2Offset = 0, int texCoord3Offset = 0,
				  int texCoord4Offset = 0, int texCoord5Offset = 0, int texCoord6Offset = 0, int texCoord7Offset = 0,
				  int size = 0); // if size == 0, then it's computed automatically from components' nums
		void free();

		int getSize();
	};
}

// ----------------------------------------------------------------------------

#endif
