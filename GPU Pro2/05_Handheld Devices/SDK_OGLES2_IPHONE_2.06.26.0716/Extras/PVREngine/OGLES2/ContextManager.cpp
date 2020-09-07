/******************************************************************************

 @File         ContextManager.cpp

 @Title        

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Manages contexts so that extensions can be accessed. TODO: make
               work with multiple contexts

******************************************************************************/

#include "ContextManager.h"

namespace pvrengine
{



/****************************************************************************
** Functions
****************************************************************************/

	/******************************************************************************/

	ContextManager::ContextManager():	// unlikely to use more than one context
			m_i32CurrentContext(0)
	{
	}

	/******************************************************************************/

	CPVRTgles2Ext* ContextManager::getExtensions()
	{
		return NULL;	// no extensions in OGLES2
	}

	/******************************************************************************/

	SPVRTContext* ContextManager::getCurrentContext()
	{
		return m_daContext[m_i32CurrentContext];
	}

	/******************************************************************************/

	void ContextManager::initContext()
	{
		// doesn't do anything in this API
	}

	/******************************************************************************/

	int ContextManager::addNewContext(SPVRTContext *pContext)
	{
		m_daContext.append(pContext);
		return m_daContext.getSize();
	}

	/******************************************************************************/

	void ContextManager::setCurrentContext(const int i32Context)
	{
		if(i32Context>0 && i32Context<m_daContext.getSize())
		{
			m_i32CurrentContext = i32Context;
		}
	}

}

/*****************************************************************************
 End of file (ContextManager.cpp)
*****************************************************************************/

