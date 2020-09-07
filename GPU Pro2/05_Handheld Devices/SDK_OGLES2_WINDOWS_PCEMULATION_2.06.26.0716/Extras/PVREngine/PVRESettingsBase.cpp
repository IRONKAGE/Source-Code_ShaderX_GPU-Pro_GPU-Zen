/******************************************************************************

 @File         PVRESettingsBase.cpp

 @Title        

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent/OGLES2

 @Description  API independent settings routines for PVREngine Implements
               functions from PVRESettings.h

******************************************************************************/

#include "PVRESettings.h"
#include "PVRTools.h"
#include "ContextManager.h"

namespace pvrengine
{
	

	/******************************************************************************/

	PVRESettings::PVRESettings()
	{
	}

	/******************************************************************************/

	PVRESettings::~PVRESettings()
	{
	}

	/******************************************************************************/

	void PVRESettings::Init()
	{
		// default blending value
		setBlendFunc(ePODBlendFunc_SRC_ALPHA,ePODBlendFunc_ONE_MINUS_SRC_ALPHA);

		// everything is dirty to start with os everything may be set
		// and no default state is assumed.
		// So responsibility is to the app for initialisation

		m_u32DirtyFlags=0xffffffff;
	}

	/******************************************************************************/

	bool PVRESettings::InitPrint3D(CPVRTPrint3D& sPrint3d,
		const unsigned int u32Width,
		const unsigned int u32Height,
		const bool bRotate)
	{
		return sPrint3d.SetTextures(ContextManager::inst().getCurrentContext(),u32Width,u32Height,bRotate) == PVR_SUCCESS;
	}

}

/******************************************************************************
End of file (PVRESettingsBase.cpp)
******************************************************************************/

