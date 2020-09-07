/******************************************************************************

 @File         Material.cpp

 @Title        Texture

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Mesh material for PVREngine

******************************************************************************/


#include "Material.h"
#include "ConsoleLog.h"
#include "MaterialManager.h"
#include "UniformHandler.h"
#include "Light.h"


namespace pvrengine
{
	
	bool Material::deactivateArrays()
	{
		int iNumUniforms = m_daMeshUniforms.getSize();
		for(int j = 0; j < iNumUniforms; ++j)
		{
			switch(m_daMeshUniforms[j].getSemantic())
			{	// if the material is set up correctly then this should work for all materials - mode is unimportant
			case eUsPosition:
			case eUsNormal:
			case eUsUV:
			case eUsTangent:
			case eUsBinormal:
			case eUsBoneIndex:
			case eUsBoneWeight:
				{
					glDisableVertexAttribArray(m_daMeshUniforms[j].getLocation());
				}
				break;
			default:
				{
					// do nothing
				}
			}
		}
		return true;
	}
}
/******************************************************************************
End of file (MaterialAPI.cpp)
******************************************************************************/

