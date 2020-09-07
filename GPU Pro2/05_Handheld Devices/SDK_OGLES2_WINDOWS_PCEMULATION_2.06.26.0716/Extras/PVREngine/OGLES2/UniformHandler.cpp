/******************************************************************************

 @File         UniformHandler.cpp

 @Title        Introducing the POD 3d file format

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     OGLES2 implementation of PVRESemanticHandler

 @Description  Shows how to use the pfx format

******************************************************************************/

#include "UniformHandler.h"
#include "Uniform.h"
#include "LightManager.h"
#include "Light.h"
#include "ContextManager.h"
#include "ConsoleLog.h"
#include <string.h>

namespace pvrengine
{

	/******************************************************************************/

	void UniformHandler::setContext(SPVRTContext* pContext)
	{
		m_pContext = pContext;
		m_pExtensions = NULL;
	}

	/******************************************************************************/

	void UniformHandler::CalculateMeshUniform(const Uniform& sUniform, SPODMesh *pMesh, SPODNode *pNode)
	{
		switch(sUniform.getSemantic())
		{
		case eUsPosition:
			{
				glVertexAttribPointer(sUniform.getLocation(), 3, GL_FLOAT, GL_FALSE, pMesh->sVertex.nStride, pMesh->sVertex.pData);
				glEnableVertexAttribArray(sUniform.getLocation());
			}
			break;
		case eUsNormal:
			{
				glVertexAttribPointer(sUniform.getLocation(), 3, GL_FLOAT, GL_FALSE, pMesh->sNormals.nStride, pMesh->sNormals.pData);
				glEnableVertexAttribArray(sUniform.getLocation());
			}
			break;
		case eUsTangent:
			{
				glVertexAttribPointer(sUniform.getLocation(), 3, GL_FLOAT, GL_FALSE, pMesh->sTangents.nStride, pMesh->sTangents.pData);
				glEnableVertexAttribArray(sUniform.getLocation());
			}
			break;
		case eUsBinormal:
			{
				glVertexAttribPointer(sUniform.getLocation(), 2, GL_FLOAT, GL_FALSE, pMesh->sBinormals.nStride, pMesh->sBinormals.pData);
				glEnableVertexAttribArray(sUniform.getLocation());
			}
			break;
		case eUsUV:
			{
				glVertexAttribPointer(sUniform.getLocation(), 2, GL_FLOAT, GL_FALSE, pMesh->psUVW[0].nStride, pMesh->psUVW[0].pData);
				glEnableVertexAttribArray(sUniform.getLocation());
			}
			break;
		case eUsBoneIndex:
			{
				glVertexAttribPointer(sUniform.getLocation(), pMesh->sBoneIdx.n, GL_UNSIGNED_BYTE, GL_FALSE, pMesh->sBoneIdx.nStride, pMesh->sBoneIdx.pData);
				glEnableVertexAttribArray(sUniform.getLocation());
			}
			break;
		case eUsBoneWeight:
			{
				glVertexAttribPointer(sUniform.getLocation(), pMesh->sBoneWeight.n, GL_FLOAT, GL_FALSE, pMesh->sBoneWeight.nStride, pMesh->sBoneWeight.pData);
				glEnableVertexAttribArray(sUniform.getLocation());
			}
			break;
		case eUsWORLD:
			{
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, m_mWorld.f);
			}
			break;
		case eUsWORLDI:
			{
				PVRTMat4 mWorldI;
				mWorldI =  m_mWorld.inverse();
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, mWorldI.f);
			}
			break;
		case eUsWORLDIT:
			{
				PVRTMat3 mWorldIT;
				mWorldIT = m_mWorld.inverse().transpose();
				glUniformMatrix3fv(sUniform.getLocation(), 1, GL_FALSE, mWorldIT.f);
			}
			break;
		case eUsWORLDVIEW:
			{
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, m_mWorldView.f);
			}
			break;
		case eUsWORLDVIEWI:
			{
				PVRTMat4 mWorldViewI;
				mWorldViewI = m_mWorldView.inverse();
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, mWorldViewI.f);
			}
			break;
		case eUsWORLDVIEWIT:
			{
				PVRTMat3 mWorldViewIT;
				mWorldViewIT = m_mWorldView.inverse().transpose();
				glUniformMatrix3fv(sUniform.getLocation(), 1, GL_FALSE, mWorldViewIT.f);
			}
			break;
		case eUsWORLDVIEWPROJECTION:
			{
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, m_mWorldViewProjection.f);
			}
			break;
		case eUsWORLDVIEWPROJECTIONI:
			{
				PVRTMat4 mWorldViewProjectionI = (m_mProjection * m_mWorldView ).inverse();
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, mWorldViewProjectionI.f);
			}
			break;
		case eUsWORLDVIEWPROJECTIONIT:
			{
				PVRTMat3 mWorldViewProjectionIT = (m_mProjection * m_mWorldView).inverse().transpose();
				glUniformMatrix3fv(sUniform.getLocation(), 1, GL_FALSE, mWorldViewProjectionIT.f);
			}
			break;
		case eUsLIGHTPOSMODEL:
			{
				// Passes the light position in eye space to the shader
				Light* pLight = m_pLightManager->get(sUniform.getIdx());
				switch(pLight->getType())
				{
				case eLightTypePoint:
					{
						PVRTVec4 vLightPosModel = m_mWorld.inverse() * ((LightPoint*)pLight)->getPositionPVRTVec4() ;
						glUniform3f(sUniform.getLocation(),
							vLightPosModel.x,
							vLightPosModel.y,
							vLightPosModel.z);
					}
					break;
				case eLightTypePODPoint:
					{
						PVRTVec4 vLightPosModel = m_mWorld.inverse() * ((LightPODPoint*)pLight)->getPositionPVRTVec4() ;
						glUniform3f(sUniform.getLocation(),
							vLightPosModel.x,
							vLightPosModel.y,
							vLightPosModel.z);
					}
					break;
				default:
					{	// hack for directional lights
						// take the light direction and multiply it by a really big negative number
						// if you hit this code then the types of your lights do not match the types expected by your shaders
						PVRTVec4 vLightPosModel = (((LightDirectional*)pLight)->getDirectionPVRTVec4()*c_fFarDistance) ;
						vLightPosModel.w = f2vt(1.0f);
						vLightPosModel = m_mWorld * vLightPosModel;
						glUniform3f(sUniform.getLocation(),
							vLightPosModel.x,
							vLightPosModel.y,
							vLightPosModel.z);
					}
				}
			}
			break;
		case eUsOBJECT:
			{
				// Scale
				PVRTMat4 mObject = m_psScene->GetScalingMatrix(*pNode);
				// Rotation
				mObject = m_psScene->GetRotationMatrix(*pNode) * mObject;
				// Translation
				mObject = m_psScene->GetTranslationMatrix(*pNode) * mObject;

				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, mObject.f);
			}
			break;
		case eUsOBJECTI:
			{
				if(!getFlag(eUsOBJECT))
				{
					// Scale
					m_mObject = m_psScene->GetScalingMatrix(*pNode);
					// Rotation
					m_mObject = m_psScene->GetRotationMatrix(*pNode) * m_mObject;
					// Translation
					m_mObject = (m_psScene->GetTranslationMatrix(*pNode) * m_mObject);
					setFlag(eUsOBJECT);
				}
				m_mObjectI = m_mObject.inverse();

				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, m_mObjectI.f);
			}
			break;
		case eUsOBJECTIT:
			{
				if(!getFlag(eUsOBJECTI))
				{
					if(!getFlag(eUsOBJECT))
					{
						// Scale
						m_mObject = m_psScene->GetScalingMatrix(*pNode);
						// Rotation
						m_mObject = m_psScene->GetRotationMatrix(*pNode) * m_mObject;
						// Translation
						m_mObject = (m_psScene->GetTranslationMatrix(*pNode) * m_mObject);
						setFlag(eUsOBJECT);
					}
					m_mObjectI = m_mObject.inverse();
					setFlag(eUsOBJECTI);
				}

				m_mObjectIT = PVRTMat3(m_mObjectI).transpose();

				glUniformMatrix3fv(sUniform.getLocation(), 1, GL_FALSE, m_mObjectIT.f);
			}
			break;
		case eUsLIGHTDIRMODEL:
			{
				Light* pLight = m_pLightManager->get(sUniform.getIdx());
				switch(pLight->getType())
				{
				case eLightTypeDirectional:
					{
						// Passes the light direction in model space to the shader
						PVRTVec4 vLightDirectionModel,
							vLightDirection =((LightDirectional*)pLight)->getDirectionPVRTVec4();
						vLightDirectionModel = m_mWorld.inverse() * vLightDirection ;
						glUniform3f(sUniform.getLocation(), vLightDirectionModel.x, vLightDirectionModel.y, vLightDirectionModel.z);
					}
				case eLightTypePODDirectional:
					{
						// Passes the light direction in model space to the shader
						PVRTVec4 vLightDirectionModel,
							vLightDirection =((LightPODDirectional*)pLight)->getDirectionPVRTVec4();
						vLightDirectionModel = m_mWorld.inverse() * vLightDirection ;
						glUniform3f(sUniform.getLocation(), vLightDirectionModel.x, vLightDirectionModel.y, vLightDirectionModel.z);
					}
				default:
					{	// hack for point lights
						// calculate vector between light position and mesh

						//TODO: get hold of the nice centre point I calculated for all these meshes and use this point
					}
				}
			}
			break;
		case eUsEYEPOSMODEL:
			{	
				m_vEyePositionModel = m_mWorld.inverse() * PVRTVec4(m_vEyePositionWorld,VERTTYPE(1.0f));
				glUniform3f(sUniform.getLocation(), m_vEyePositionModel.x, m_vEyePositionModel.y, m_vEyePositionModel.z);
			}
			break;
		default:
			{	// something went wrong
				ConsoleLog::inst().log("Error: non-mesh uniform being interpreted as mesh uniform\n");
				return;
			}
		}
	}

	/******************************************************************************/

	void UniformHandler::BindFrameUniform(const Uniform& sUniform)
	{
		EUniformSemantic eSemantic = sUniform.getSemantic();
		unsigned int index = sUniform.getIdx();
		_ASSERT(getFlag(eSemantic,index));

		switch(eSemantic)
		{
		case eUsVIEW:
			{
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, m_mView.f);
			}
			break;
		case eUsVIEWI:
			{
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, m_mViewI.f);
			}
			break;
		case eUsVIEWIT:
			{
				glUniformMatrix3fv(sUniform.getLocation(), 1, GL_FALSE, m_mViewIT.f);
			}
			break;
		case eUsPROJECTION:
			{
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, m_mProjection.f);
			}
			break;
		case eUsPROJECTIONI:
			{
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, m_mProjectionI.f);
			}
			break;
		case eUsPROJECTIONIT:
			{
				glUniformMatrix3fv(sUniform.getLocation(), 1, GL_FALSE, m_mProjectionIT.f);
			}
			break;
		case eUsVIEWPROJECTION:
			{
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, m_mViewProjection.f);
			}
			break;
		case eUsVIEWPROJECTIONI:
			{
				glUniformMatrix4fv(sUniform.getLocation(), 1, GL_FALSE, m_mViewProjectionI.f);
			}
			break;
		case eUsVIEWPROJECTIONIT:
			{
				glUniformMatrix3fv(sUniform.getLocation(), 1, GL_FALSE, m_mViewProjectionIT.f);
			}
			break;
		case eUsLIGHTCOLOR:
			{
				glUniform3f(sUniform.getLocation(), m_pfLightColor[index][0],
					m_pfLightColor[index][1], m_pfLightColor[index][2]);
			}
			break;
		case eUsLIGHTPOSWORLD:
			{
				glUniform3f(sUniform.getLocation(), m_vLightPosWorld[index].x,
					m_vLightPosWorld[index].y, m_vLightPosWorld[index].z);
			}
			break;
		case eUsLIGHTPOSEYE:
			{
				glUniform3f(sUniform.getLocation(), m_vLightPosEye[index].x,
					m_vLightPosEye[index].y, m_vLightPosEye[index].z);
			}
			break;
		case eUsLIGHTDIRWORLD:
			{
				glUniform3f(sUniform.getLocation(), m_vLightDirWorld[index].x,
					m_vLightDirWorld[index].y, m_vLightDirWorld[index].z);
			}
			break;
		case eUsLIGHTDIREYE:
			{
				glUniform3f(sUniform.getLocation(), m_vLightDirEye[index].x,
					m_vLightDirEye[index].y, m_vLightDirEye[index].z);
			}
			break;
		case eUsEYEPOSWORLD:
			{
				glUniform3f(sUniform.getLocation(),
					m_vEyePositionWorld.x, m_vEyePositionWorld.y, m_vEyePositionWorld.z);
			}
			break;
		case eUsANIMATION:
			{
				glUniform1f(sUniform.getLocation(), m_fAnimation);
			}
			break;
		default:
			{	// something went wrong
				ConsoleLog::inst().log("Error: non-frame uniform being interpreted as frame uniform\n");
				return;
			}
		}
	}

	/******************************************************************************/

	void UniformHandler::CalculateMaterialUniform(const Uniform* pUniform, Material* pMaterial)
	{
		switch(pUniform->getSemantic())
		{
		case eUsMATERIALOPACITY:
			{
				glUniform1f(pUniform->getLocation(),
					pMaterial->getOpacity());
			}
			break;
		case eUsMATERIALSHININESS:
			{
				glUniform1f(pUniform->getLocation(),
					pMaterial->getShininess());
			}
			break;
		case eUsMATERIALCOLORAMBIENT:
			{
				PVRTVec3 vColour = pMaterial->getAmbient();
				glUniform3f(pUniform->getLocation(), vColour.x, vColour.y, vColour.z);
			}
			break;
		case eUsMATERIALCOLORDIFFUSE:
			{
				PVRTVec3 vColour = pMaterial->getDiffuse();
				glUniform3f(pUniform->getLocation(), vColour.x, vColour.y, vColour.z);
			}
			break;
		case eUsMATERIALCOLORSPECULAR:
			{
				PVRTVec3 vColour = pMaterial->getSpecular();
				glUniform3f(pUniform->getLocation(), vColour.x, vColour.y, vColour.z);
			}
			break;
		case eUsTEXTURE:
			{
				// Set the sampler variable to the texture unit
				glUniform1i(pUniform->getLocation(), pUniform->getIdx());
			}
			break;
		default:
			{	// something went wrong
				ConsoleLog::inst().log("Error: non-material uniform being interpreted as material uniform\n");
				return;
			}
		}
	}

}

/******************************************************************************
End of file (UniformHandlerAPI.cpp)
******************************************************************************/

