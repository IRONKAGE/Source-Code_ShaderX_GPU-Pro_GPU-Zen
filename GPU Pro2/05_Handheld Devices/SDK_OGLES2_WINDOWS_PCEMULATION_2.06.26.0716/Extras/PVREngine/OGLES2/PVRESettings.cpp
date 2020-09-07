/******************************************************************************

 @File         PVRESettings.cpp

 @Title        

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     OGLES2 implementation of PVRESettings

 @Description  Settings class for the PVREngine

******************************************************************************/
#include "PVRESettings.h"
#include "ContextManager.h"


namespace pvrengine
{

	/******************************************************************************/
	
	void PVRESettings::setBlend(const bool bBlend)
	{
		if(bBlend)
			glEnable(GL_BLEND);
		else
			glDisable(GL_BLEND);
	}

	/******************************************************************************/
	
	void PVRESettings::setBlendFunc(const EPODBlendFunc eSource, const EPODBlendFunc eDest)
	{
		glBlendFunc(eSource, eDest);
	}
	
	/******************************************************************************/
	
	void PVRESettings::setBlendFuncSeparate(const EPODBlendFunc eSourceRGB,
		const EPODBlendFunc eDestRGB,
		const EPODBlendFunc eSourceA,
		const EPODBlendFunc eDestA)
	{
		if(isDirty(eStateBlendFunction) ||
			((eSourceRGB!=m_eBlendSourceRGB) ||(eDestRGB!=m_eBlendDestRGB) ||
			(eSourceA!=m_eBlendSourceA) || (eDestA!=m_eBlendDestA)))
		{
			m_eBlendSourceRGB = eSourceRGB;
			m_eBlendDestRGB = eDestRGB;
			m_eBlendSourceA = eSourceA;
			m_eBlendDestA = eDestA;
			glBlendFuncSeparate(eSourceRGB,eDestRGB,eSourceA,eDestA);
			setClean(eStateBlendFunction);
		}
	}

	/******************************************************************************/

	void PVRESettings::setBlendOperation(const EPODBlendOp eOp)
	{
		if(isDirty(eStateBlendOp) || (eOp!=m_eBlendOperationRGB || eOp!=m_eBlendOperationA))
		{
			m_eBlendOperationRGB = eOp;
			m_eBlendOperationA = eOp;
			glBlendEquation(eOp);
			setClean(eStateBlendOp);
		}
	}

	/******************************************************************************/

	void PVRESettings::setBlendOperationSeparate(const EPODBlendOp eOpRGB,
			const EPODBlendOp eOpA)
	{
		if(isDirty(eStateBlendOp) || (eOpRGB!=m_eBlendOperationRGB || eOpA!=m_eBlendOperationA))
		{
			m_eBlendOperationRGB = eOpRGB;
			m_eBlendOperationA = eOpA;
			glBlendEquationSeparate(eOpRGB,eOpA);
			setClean(eStateBlendOp);
		}
	}

	/******************************************************************************/

	void PVRESettings::setBlendColour(const PVRTVec4 v4BlendColour)
	{
		if(isDirty(eStateBlendColour) || v4BlendColour!=m_v4BlendColour)
		{
			m_v4BlendColour = v4BlendColour;
			glBlendColor(v4BlendColour.x,v4BlendColour.y,v4BlendColour.z,v4BlendColour.w);
			setClean(eStateBlendColour);
		}
	}

	/******************************************************************************/
	
	void PVRESettings::setBackColour(const PVRTVec4 v4BackColour)
	{	// this is the base function used by the other versions
		if(isDirty(eStateBackColour) ||
			(m_v4BackColour != v4BackColour))
		{	//only store and set if different from stored colour or if unset/dirty
			m_v4BackColour = v4BackColour;

			glClearColor(m_v4BackColour.x,m_v4BackColour.y,m_v4BackColour.z,m_v4BackColour.w);
			setClean(eStateBackColour);
		}
		// else same colour and not dirty so don't waste a GL call

	}

	/******************************************************************************/

	void PVRESettings::setBackColour(const unsigned int u32BackColour)
	{
		setBackColour(PVRTVec4((VERTTYPE)(u32BackColour>>24&0xff)/255.0f,
			(VERTTYPE)(u32BackColour>>16&0xff)/255.0f,
			(VERTTYPE)(u32BackColour>>8&0xff)/255.0f,
			(VERTTYPE)(u32BackColour&0xff)/255.0f));
	}

	/******************************************************************************/

	void PVRESettings::setBackColour(const VERTTYPE fRed, const VERTTYPE fGreen, const VERTTYPE fBlue, const VERTTYPE fAlpha)
	{
		setBackColour(PVRTVec4(fRed, fGreen, fBlue, fAlpha));
	}
	/******************************************************************************/

	void PVRESettings::setBackColour(const VERTTYPE fRed, const VERTTYPE fGreen, const VERTTYPE fBlue)
	{
		setBackColour(PVRTVec4(fRed, fGreen, fBlue, 1.0f));
	}

	/******************************************************************************/

	void PVRESettings::setBackColour(const unsigned int u32Red,
									 const unsigned int u32Green, 
									 const unsigned int u32Blue,
									 const unsigned int u32Alpha)
	{
		setBackColour(PVRTVec4((float(u32Red))/255.0f,(float(u32Green))/255.0f,(float(u32Blue))/255.0f,(float(u32Alpha))/255.0f));
	}

	/******************************************************************************/

	void PVRESettings::setBackColour(const unsigned int u32Red,
									 const unsigned int u32Green, 
									 const unsigned int u32Blue)
	{
		setBackColour(PVRTVec4((float(u32Red))/255.0f,(float(u32Green))/255.0f,(float(u32Blue))/255.0f,1.0f));
	}

	/******************************************************************************/

	void PVRESettings::setClearFlags(unsigned int u32ClearFlags)
	{
		// this one isn't actually set in GL so just cache
		m_u32ClearFlags = u32ClearFlags;
	}

	/******************************************************************************/

	unsigned int PVRESettings::getClearFlags()
	{
		// this one isn't actually set in GL so just return cache
		return m_u32ClearFlags;
	}

	/******************************************************************************/

	void PVRESettings::Clear()
	{
		glClear(m_u32ClearFlags);
	}

	/******************************************************************************/

	void PVRESettings::setDepthTest(const bool bDepth)
	{
		if(isDirty(eStateDepthTest) || m_bDepthTest!=bDepth)
		{
			m_bDepthTest = bDepth;
			if(bDepth)// Enables depth test using the z-buffer
				glEnable(GL_DEPTH_TEST);
			else
				glDisable(GL_DEPTH_TEST);
			setClean(eStateDepthTest);
		}
		// else not dirty and values same so do nothing

	}

	/******************************************************************************/

	void PVRESettings::setDepthWrite(const bool bDepthWrite)
	{
		if(isDirty(eStateDepthWrite) || (m_bDepthWrite!=bDepthWrite))
		{
			m_bDepthWrite = bDepthWrite;
			glDepthMask(bDepthWrite);
			setClean(eStateDepthWrite);
		}
	}

	/******************************************************************************/

	void PVRESettings::setCullFace(const bool bCullFace)
	{
		if(isDirty(eStateCullFace) ||
			(m_bCullFace!=bCullFace))
		{
			m_bCullFace = bCullFace;
			if(bCullFace)
				glEnable(GL_CULL_FACE);
			else
				glDisable(GL_CULL_FACE);
			setClean(eStateCullFace);
		}
	}

	/******************************************************************************/

	void PVRESettings::setCullMode(const ECullFaceMode eMode)
	{
		if(isDirty(eStateCullFaceMode) ||
			(m_eCullFaceMode!=eMode))
		{
			m_eCullFaceMode = eMode;
			glCullFace(eMode);
			setClean(eStateCullFaceMode);
		}
	}

	/******************************************************************************/

	CPVRTString PVRESettings::getAPIName()
	{
		return CPVRTString("OpenGL|ES2");
	}



}

/******************************************************************************
End of file (PVRESettings.cpp)
******************************************************************************/

