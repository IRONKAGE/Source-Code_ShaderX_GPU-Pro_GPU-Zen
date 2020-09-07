/******************************************************************************

 @File         LightManager.cpp

 @Title        Introducing the POD 3d file format

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Manages lights

******************************************************************************/

#include "LightManager.h"
#include "Light.h"
#include "Globals.h"


namespace pvrengine
{
	



	/******************************************************************************/

	void	LightManager::shineLights()
	{
		for(int i=0;i<m_daElements.getSize();i++)
		{
			m_daElements[i]->shineLight(i);
		}
	}


}

/******************************************************************************
End of file (LightManager.cpp)
******************************************************************************/

