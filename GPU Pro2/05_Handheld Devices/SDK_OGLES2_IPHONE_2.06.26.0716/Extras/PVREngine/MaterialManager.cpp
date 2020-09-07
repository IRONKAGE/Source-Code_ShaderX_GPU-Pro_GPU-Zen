/******************************************************************************

 @File         MaterialManager.cpp

 @Title        Introducing the POD 3d file format

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Shows how to use the pfx format

******************************************************************************/

#include "PVRTools.h"
#include "MaterialManager.h"
#include "Globals.h"
#include "Material.h"
#include "ConsoleLog.h"

#define NONE_STRING "*** None ***";

namespace pvrengine
{

	

	/******************************************************************************/

	MaterialManager::MaterialManager()
	{
		internalInit();
	}

	/******************************************************************************/

	MaterialManager::MaterialManager(int i32MaxMaterials):
		Manager<Material>(i32MaxMaterials)
	{
		internalInit();
	}

	/******************************************************************************/

	MaterialManager::~MaterialManager()
	{
		for(int i=0;i<m_daElements.getSize();++i)
		{	// check if material is already in manager
			for(int j=i+1;j<m_daElements.getSize();++j)
			{
				if(m_daElements[i]==m_daElements[j]
				|| m_daElements[i]==m_pFlatMaterial)
				{
					m_daElements[i]=NULL;
					break;
				}
			}
		}

		for(int i=0;i<m_daElements.getSize();++i)
		{
			if(m_daElements[i]!=m_pFlatMaterial)
				PVRDELETE(m_daElements[i]);
		}

		PVRDELETE(m_pFlatMaterial);
	}

	/******************************************************************************/

	bool MaterialManager::init(UniformHandler* pUniformHandler)
	{
		// init flat material
		_ASSERT(m_pFlatMaterial==NULL);

		m_pFlatMaterial = new Material(eFlat, pUniformHandler, this);

		return true;
	}
	/******************************************************************************/

	void MaterialManager::internalInit()
	{
		m_pActiveMaterial = NULL;
		m_pFlatMaterial = NULL;
	}

	/******************************************************************************/

	Material* MaterialManager::LoadMaterial(const CPVRTString& strEffectPath,
		const CPVRTString& strTexturePath,
		const SPODMaterial& sPODMaterial,
		const SPODTexture& sPODTexture,
		UniformHandler* pUniformHandler)
	{
		CPVRTString strPFXFilename = NONE_STRING;
		if(sPODMaterial.pszEffectFile)
		{
			strPFXFilename = strEffectPath;
			((strPFXFilename+="/")+=sPODMaterial.pszEffectFile);
		}
		else
		{
			strPFXFilename = NONE_STRING;
		}

		// resolve texture file name/path
		CPVRTString strTextureFile = strTexturePath;
		if(!sPODMaterial.pszEffectFile)
		{
			if(sPODMaterial.nIdxTexDiffuse!=-1 && sPODTexture.pszName)
			{	//TODO: proper path separator
				(strTextureFile+="/") += sPODTexture.pszName;
			}
		}

		for(int i=0;i<m_daElements.getSize();++i)
		{	// check if material is already in manager
			// check
			if(m_daElements[i]->getEffectFileName().compare(strPFXFilename)==0
				&&m_daElements[i]->getEffectName().compare(sPODMaterial.pszEffectName)==0
				&&m_daElements[i]->getName().compare(sPODMaterial.pszName)==0
				&&m_daElements[i]->getTextureFileName().compare(strTextureFile)==0)
			{
				ConsoleLog::inst().log("\nMaterial %s already in manager.",sPODMaterial.pszName);
				m_daElements.append(m_daElements[i]);

				return m_daElements[i];
			}
		}

		// actually make/load material - and attempt error handle
		Material* pNewMaterial;
		if(sPODMaterial.pszEffectFile)
		{	// with effect -> m_daElements.getSize() gives unique id to a material
			pNewMaterial = new Material(m_daElements.getSize(),strPFXFilename,strTexturePath,sPODMaterial, pUniformHandler, this);
		}
		else
		{	// no effect
			pNewMaterial = new Material(m_daElements.getSize(),strTextureFile,sPODMaterial, pUniformHandler, this);
		}

		if((!pNewMaterial) || !pNewMaterial->getValid())
		{
			ConsoleLog::inst().log("Loading material failed. Replacing with flat material\n");
			PVRDELETE(pNewMaterial);
			m_daElements.append(m_pFlatMaterial);
			return(m_pFlatMaterial);
		}

		m_daElements.append(pNewMaterial);

		return pNewMaterial;
	}

	/******************************************************************************/

	Material* MaterialManager::getMaterial(const unsigned int u32Id)
	{
		return m_daElements[u32Id];
	}


	/******************************************************************************/

	void MaterialManager::ReportActiveMaterial(Material* pNewMaterial)
	{
		if(m_pActiveMaterial)
			m_pActiveMaterial->Deactivate();	// only one active material at a time
		m_pActiveMaterial = pNewMaterial;
	}

	/******************************************************************************/

	Material*	MaterialManager::getFlatMaterial()
	{
		return m_pFlatMaterial;
	}

}

/******************************************************************************
End of file (MaterialManager.cpp)
******************************************************************************/

