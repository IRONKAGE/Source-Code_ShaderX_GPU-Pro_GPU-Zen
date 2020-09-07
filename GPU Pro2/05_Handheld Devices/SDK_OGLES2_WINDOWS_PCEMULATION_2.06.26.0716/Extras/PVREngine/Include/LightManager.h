/******************************************************************************

 @File         LightManager.h

 @Title        A simple light manager for use with PVREngine

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A class for holding information about lights

******************************************************************************/

#ifndef LIGHTMANAGER_H
#define LIGHTMANAGER_H

/******************************************************************************
Includes
******************************************************************************/

#include "Manager.h"
#include "Light.h"

namespace pvrengine
{

	/*!***************************************************************************
	* @Class LightManager
	* @Brief A class for holding information about lights
	* @Description A class for holding information about lights
	*****************************************************************************/
	class LightManager : public Manager<Light>
	{

	public:

		/*!***************************************************************************
		@Function			shineLights
		@Description		very unsubtle function (atm) to initialise the lights
		in a scene
		*****************************************************************************/
		void	shineLights();


	private:

	};

}
#endif // LIGHTMANAGER_H

/******************************************************************************
End of file (LightManager.h)
******************************************************************************/

