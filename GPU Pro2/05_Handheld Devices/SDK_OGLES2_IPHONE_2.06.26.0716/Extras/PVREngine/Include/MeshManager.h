/******************************************************************************

 @File         MeshManager.h

 @Title        A simple nesh manager for use with PVREngine

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A class for holding information about meshes as they are loaded so
               that duplicate meshes are not kept in memory and so the things can
               be disposed of easily at the end of execution.

******************************************************************************/

#ifndef MESHMANAGER_H
#define MESHMANAGER_H

/******************************************************************************
Includes
******************************************************************************/

#include "PVRTools.h"
#include "Manager.h"
#include "Mesh.h"

namespace pvrengine
{
	/*!***************************************************************************
	* @Class MeshManager
	* @Brief A class for managing meshes.
	* @Description A class for managing meshes.
	*****************************************************************************/
	class MeshManager:public Manager<Mesh>
	{
	public:
		/*!***************************************************************************
		@Function			setDrawMode
		@Input				eDrawMode	the desired drawmode
		@Description		Blanket sets the drawng modes for all the meshes held
		within the manager.
		*****************************************************************************/
		void setDrawMode(const EDrawMode eDrawMode);

	private:

	};

}
#endif // MESHMANAGER_H

/******************************************************************************
End of file (MeshManager.h)
******************************************************************************/

