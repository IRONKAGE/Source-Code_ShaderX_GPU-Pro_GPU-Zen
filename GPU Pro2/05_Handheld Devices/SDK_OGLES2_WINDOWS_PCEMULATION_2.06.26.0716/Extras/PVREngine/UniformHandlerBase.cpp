/******************************************************************************

 @File         UniformHandlerBase.cpp

 @Title        Introducing the POD 3d file format

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     OGLES2 implementation of PVRESemanticHandler

 @Description  Shows how to use the pfx format

******************************************************************************/

#include "PVRTools.h"
#include "UniformHandler.h"
#include "ConsoleLog.h"

namespace pvrengine
{

	/******************************************************************************/

	void UniformHandler::setView(const PVRTVec3& vFrom, const PVRTVec3& vTo, const PVRTVec3& vUp)
	{
		m_vEyePositionWorld = vFrom;
		m_mView = PVRTMat4::LookAtRH(vFrom, vTo, vUp);
		m_mViewProjection = m_mProjection *m_mView;
	}

	/******************************************************************************/

	SPVRTContext* UniformHandler::getContext()
	{
		return m_pContext;
	}

	/******************************************************************************/

	void UniformHandler::setProjection(VERTTYPE fFOV,
		VERTTYPE fAspectRatio,
		VERTTYPE fNear,
		VERTTYPE fFar,
		bool bRotate)
	{
		m_fFOV = fFOV;
		m_fAspectRatio = fAspectRatio;
		m_fNear = fNear;
		m_fFar = fFar;
		m_bRotate = bRotate;
		m_mProjection = PVRTMat4::PerspectiveFovRH(fFOV,
			fAspectRatio,
			fNear,
			fFar,
			PVRTMat4::OGL,
			bRotate);
	}

		/******************************************************************************/

	void UniformHandler::CalculateFrameUniform(const Uniform& sUniform)
	{
		unsigned int u32Semantic = sUniform.getSemantic();
		unsigned int index = sUniform.getIdx();
		switch(sUniform.getSemantic())
		{
		case eUsVIEW:
			{
				// calculation not required
			}
			break;
		case eUsVIEWI:
			{
				m_mViewI = m_mView.inverse();
			}
			break;
		case eUsVIEWIT:
			{
				if(!getFlag(eUsVIEWI))
				{
					m_mViewI = m_mView.inverse();
					setFlag(eUsVIEWI);
				}
				m_mViewIT = m_mViewI.transpose();
			}
			break;
		case eUsPROJECTION:
			{
				// calculation not required
			}
			break;
		case eUsPROJECTIONI:
			{
				m_mProjectionI = m_mProjection.inverse();
			}
			break;
		case eUsPROJECTIONIT:
			{
				if(!getFlag(eUsPROJECTIONI))
				{
					m_mProjectionI = m_mProjection.inverse();
					setFlag(eUsPROJECTIONI);
				}
				m_mProjectionIT = m_mProjectionI.transpose();
			}
			break;
		case eUsVIEWPROJECTION:
			{
				// calculation not required
			}
			break;
		case eUsVIEWPROJECTIONI:
			{
				if(!getFlag(eUsVIEWPROJECTION))
				{
					m_mViewProjection = m_mProjection * m_mView;
					setFlag(eUsVIEWPROJECTION);
				}
				m_mViewProjectionI = m_mProjection.inverse();
			}
			break;
		case eUsVIEWPROJECTIONIT:
			{
				if(!getFlag(eUsVIEWPROJECTIONI))
				{
					if(!getFlag(eUsVIEWPROJECTION))
					{
						m_mViewProjection = m_mProjection * m_mView;
						setFlag(eUsVIEWPROJECTION);
					}
					m_mViewProjectionI = m_mProjection.inverse();
					setFlag(eUsVIEWPROJECTIONI);
				}
				m_mViewProjectionIT = m_mViewProjectionI.transpose();
			}
			break;
		case eUsLIGHTCOLOR:
			{
				m_pfLightColor[sUniform.getIdx()] = m_psScene->pLight[sUniform.getIdx()].pfColour;
			}
			break;

		case eUsLIGHTPOSWORLD:
			{
				Light* pLight = m_pLightManager->get(sUniform.getIdx());
				switch(pLight->getType())
				{
				case eLightTypePoint:
					{
						m_vLightPosWorld[index] = ((LightPoint*)pLight)->getPositionPVRTVec4();
						break;
					}
				case eLightTypePODPoint:
					{
						m_vLightPosWorld[index] = ((LightPODPoint*)pLight)->getPositionPVRTVec4();
						break;
					}
				case eLightTypeDirectional:
					{	// hack for directional lights
						// take the light direction and multiply it by a really big negative number
						m_vLightDirWorld[index] = ((LightDirectional*)pLight)->getDirectionPVRTVec4();
						setFlag(eUsLIGHTDIRWORLD,index);

						m_vLightPosWorld[index] = m_vLightDirWorld[index]*c_fFarDistance;
						m_vLightPosWorld[index].w = f2vt(1.0f);

					}
				case eLightTypePODDirectional:
					{	// hack for directional lights
						// take the light direction and multiply it by a really big negative number
						m_vLightDirWorld[index] = ((LightPODDirectional*)pLight)->getDirectionPVRTVec4();
						setFlag(eUsLIGHTDIRWORLD,index);

						m_vLightPosWorld[index] = m_vLightDirWorld[index]*c_fFarDistance;
						m_vLightPosWorld[index].w = f2vt(1.0f);

					}
				default:
					{
						ConsoleLog::inst().log((char*)"Unsupported light type for LIGHTPOSWORLD semantic.\n");
						return;
					}
				}
				break;
			}
		case eUsLIGHTPOSEYE:
			{
				Light* pLight = m_pLightManager->get(sUniform.getIdx());
				switch(pLight->getType())
				{
				case eLightTypePoint:
					{
						m_vLightPosWorld[index] = ((LightPoint*)pLight)->getPositionPVRTVec4();
						break;
					}
				case eLightTypePODPoint:
					{
						m_vLightPosWorld[index] = ((LightPODPoint*)pLight)->getPositionPVRTVec4();
						break;
					}
				case eLightTypeDirectional:
					{	// hack for directional lights
						// take the light direction and multiply it by a really big negative number
						m_vLightDirWorld[index] = ((LightDirectional*)pLight)->getDirectionPVRTVec4();
						setFlag(eUsLIGHTDIRWORLD,index);

						m_vLightPosWorld[index] = m_vLightDirWorld[index]*c_fFarDistance;
						m_vLightPosWorld[index].w = f2vt(1.0f);

					}
				case eLightTypePODDirectional:
					{	// hack for directional lights
						// take the light direction and multiply it by a really big negative number
						m_vLightDirWorld[index] = ((LightPODDirectional*)pLight)->getDirectionPVRTVec4();
						setFlag(eUsLIGHTDIRWORLD,index);

						m_vLightPosWorld[index] = m_vLightDirWorld[index]*c_fFarDistance;
						m_vLightPosWorld[index].w = f2vt(1.0f);

					}
				default:
					{
						ConsoleLog::inst().log((char*)"Unsupported light type for LIGHTPOSEYE semantic.\n");
						return;
					}
				}
				setFlag(eUsLIGHTPOSWORLD, index);	// mark that this has been calculated
				// store light position in eye space 
				m_vLightPosEye[index] =  m_mView * m_vLightPosWorld[index];
			}
			break;
		case eUsLIGHTDIRWORLD:
			{
				// gets the light direction.
				Light* pLight = m_pLightManager->get(sUniform.getIdx());
				switch(pLight->getType())
				{
				case eLightTypeDirectional:
					{
						m_vLightDirWorld[index] = ((LightDirectional*)pLight)->getDirectionPVRTVec4();
						break;
					}
				case eLightTypePODDirectional:
					{
						m_vLightDirWorld[index] = ((LightPODDirectional*)pLight)->getDirectionPVRTVec4();
						break;
					}
				default:
					{	// hack for point lights
						// TODO: get centre from mesh
						// which may be difficult for the entire frame...
						ConsoleLog::inst().log((char*)"Unsupported light type for LIGHTDIRWORLD semantic.\n");
						return;
					}
				}
			}
			break;
		case eUsLIGHTDIREYE:
			{
				// gets the light direction.
				Light* pLight = m_pLightManager->get(sUniform.getIdx());
				switch(pLight->getType())
				{
				case eLightTypeDirectional:
					{
						m_vLightDirWorld[index] = ((LightDirectional*)pLight)->getDirectionPVRTVec4();
						setFlag(eUsLIGHTDIRWORLD,index);
					}
				case eLightTypePODDirectional:
					{
						m_vLightDirWorld[index] = ((LightPODDirectional*)pLight)->getDirectionPVRTVec4();
						setFlag(eUsLIGHTDIRWORLD,index);
						break;
					}
				default:
					{	// hack for point lights
						// TODO: get centre from mesh
						// which may be difficult for the entire frame...
						ConsoleLog::inst().log((char*)"Unsupported light type for LIGHTDIREYE semantic.\n");
						return;
					}
				}
				// Passes the light direction in eye space to the shader
				m_vLightDirEye[index] = m_vLightDirWorld[index] * m_mView;
			}
			break;
		case eUsEYEPOSWORLD:
			// calculation not required
			break;
		case eUsANIMATION:
			{
				if(m_psScene->nNumFrame > 0.0 && m_fFrame > 0.0)
				{
					// Float in the range 0..1: contains this objects distance through its animation.
					m_fAnimation = (float)m_fFrame / (float)m_psScene->nNumFrame;
				}
				else
				{
					m_fAnimation=0.0f;
				}
			}
			break;
		default:
			{	// something went wrong
				ConsoleLog::inst().log("Error: non-frame uniform being interpreted as frame uniform\n");
				return;
			}
		}
		setFlag(u32Semantic,index);	// this uniform is now set - may return above if this can't be set
	}


	/******************************************************************************/

	UniformHandler::UniformHandler():m_pContext(NULL),m_pExtensions(NULL){ResetFrameUniforms();}

	/******************************************************************************/

	void UniformHandler::ResetFrameUniforms()
	{
		memset(m_pu32FrameUniformFlags,0,eNumSemantics*sizeof(unsigned int));
	}

	/******************************************************************************/

	void UniformHandler::DoFrameUniform(const Uniform& sUniform)
	{
		if(!getFlag(sUniform.getSemantic(),sUniform.getIdx()))
			CalculateFrameUniform(sUniform);
		BindFrameUniform(sUniform);
	}

	/******************************************************************************/

	void UniformHandler::setScene(CPVRTModelPOD *psScene)
	{
		m_psScene = psScene;
	}

	/******************************************************************************/

	void UniformHandler::setFrame(const float fFrame)
	{
		m_fFrame = fFrame;
	}

	/******************************************************************************/

	bool UniformHandler::isVisibleSphere(const PVRTVec3& v3Centre, const VERTTYPE fRadius)
	{
		// get in view space
		PVRTVec4 v4TransCentre = m_mWorldView * PVRTVec4(v3Centre,f2vt(1.0f));

		// find clip space coord for centre
		v4TransCentre = m_mProjection * v4TransCentre;

		VERTTYPE fRadX,fRadY;
		// scale radius according to perspective
		if(m_bRotate)
		{
		fRadX = PVRTABS(VERTTYPEMUL(fRadius,m_mProjection(0,1)));
		fRadY = PVRTABS(VERTTYPEMUL(fRadius,m_mProjection(1,0)));
		}
		else
		{
		fRadX = PVRTABS(VERTTYPEMUL(fRadius,m_mProjection(0,0)));
		fRadY = PVRTABS(VERTTYPEMUL(fRadius,m_mProjection(1,1)));
		}
		VERTTYPE fRadZ = PVRTABS(VERTTYPEMUL(fRadius,m_mProjection(2,2)));


		// check if inside frustums
		// X
		if(v4TransCentre.x+fRadX<-v4TransCentre.w)
		{	// 'right' side out to 'left' def out
			return false;
		}
		if(v4TransCentre.x-fRadX>v4TransCentre.w)
		{	// 'left' side out to 'right' def out
			return false;
		}

		// Y
		if(v4TransCentre.y+fRadY<-v4TransCentre.w)
		{	// 'up' side out to 'top' def out
			return false;
		}
		if(v4TransCentre.y-fRadY>v4TransCentre.w)
		{	// 'down' side out to 'bottom' def out
			return false;
		}

		// Z
		if(v4TransCentre.z+fRadZ<-v4TransCentre.w)
		{	// 'far' side out to 'back' def out
			return false;
		}
		if(v4TransCentre.z-fRadZ>v4TransCentre.w)
		{	// 'near' side out to 'front' def out
			return false;
		}

		return true;
	}

}

/******************************************************************************
End of file (UniformHandlerAPI.cpp)
******************************************************************************/

