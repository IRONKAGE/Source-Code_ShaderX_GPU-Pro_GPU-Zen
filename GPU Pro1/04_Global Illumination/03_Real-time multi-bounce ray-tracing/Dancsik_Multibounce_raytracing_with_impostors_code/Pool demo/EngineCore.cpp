#include "dxstdafx.h"
#include "EngineCore.h"
#include "xmlParser.h"
#include "ShadedMesh.h"
#include "Material.h"
#include "Entity.h"
#include "NodeGroup.h"
#include "SpotLight.h"
#include "RenderContext.h"
#include "Role.h"
#include "EnvironmentMappedEntity.h"
#include "RefractorMesh.h"
#include "RigidModel.h"
#include "RigidBody.h"

EngineCore::EngineCore(LPDIRECT3DDEVICE9 device) : GameInterface(device)
{
	worldRoot = NULL;
}

HRESULT EngineCore::createManagedResources()
{
	//process level file
	//load meshes, textures, construct shaded meshes, entities
	//load rigid body models, construct rigid bodies
	//set player, camera entities

	D3DXCreateFont( device,            // D3D device
                         17,               // Height
                         0,                     // Width
                         FW_BOLD,               // Weight
                         1,                     // MipLevels, 0 = autogen mipmaps
                         FALSE,                 // Italic
                         DEFAULT_CHARSET,       // CharSet
                         OUT_DEFAULT_PRECIS,    // OutputPrecision
                         DEFAULT_QUALITY,       // Quality
                         DEFAULT_PITCH | FF_DONTCARE, // PitchAndFamily
                         L"Arial",              // pFaceName
                         &Font);              // ppFont

	loadLevel(L"Media\\level.xml");

	camera.SetViewParams( &D3DXVECTOR3(-175, 54, 250),
						  &D3DXVECTOR3(0, 0, 0));
	camera.SetScalers(0.01f, 40.0f);
	camera.SetProjParams(0.7,1,0.01,3000);

	return S_OK;
}

HRESULT EngineCore::releaseManagedResources()
{
	{
		MeshDirectory::iterator i = meshDirectory.begin();
		while(i != meshDirectory.end())
		{
			i->second->Release();
			i++;
		}
	}
	{
		TextureDirectory::iterator i = textureDirectory.begin();
		while(i != textureDirectory.end())
		{
			i->second->Release();
			i++;
		}
	}
	{
		CubeTextureDirectory::iterator i = cubeTextureDirectory.begin();
		while(i != cubeTextureDirectory.end())
		{
			i->second->Release();
			i++;
		}
	}
	{
		ShadedMeshDirectory::iterator i = shadedMeshDirectory.begin();
		while(i != shadedMeshDirectory.end())
		{
			delete i->second;
			i++;
		}
	}
	{
		SpotLightDirectory::iterator i = spotLightDirectory.begin();
		while(i != spotLightDirectory.end())
		{
			delete i->second;
			i++;
		}
	}
	delete worldRoot;
	
	Font->Release();

	return S_OK;
}

HRESULT EngineCore::createDefaultResources(wchar_t* effectFileName)
{
	LPD3DXBUFFER compilationErrors;
	if(FAILED(D3DXCreateEffectFromFile(device, effectFileName, NULL, NULL, 0, NULL, &effect, &compilationErrors)))
	{
		if(compilationErrors) MessageBoxA( NULL, (LPSTR)compilationErrors->GetBufferPointer(), "Failed to load effect file!", MB_OK);
		exit(-1);
	}
	
	device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);

	worldRoot->createDefaultResources(this);

	return S_OK;
}

HRESULT EngineCore::releaseDefaultResources()
{
	effect->Release();

	worldRoot->releaseDefaultResources();

	return S_OK;
}

LPDIRECT3DCUBETEXTURE9 EngineCore::CreateCubeTexture(int size, D3DFORMAT Format)
{
	HRESULT hr;
	LPDIRECT3DCUBETEXTURE9 CubeTexture;
	V( device->CreateCubeTexture(	size, 1, D3DUSAGE_RENDERTARGET,
									Format, D3DPOOL_DEFAULT, &CubeTexture, NULL ) );
	return CubeTexture;
}

LPDIRECT3DTEXTURE9 EngineCore::CreateTexture(int width, int height, D3DFORMAT Format)
{
	HRESULT hr;
	LPDIRECT3DTEXTURE9 Texture;
	V( device->CreateTexture(	width, height, 1, D3DUSAGE_RENDERTARGET, 
								Format, D3DPOOL_DEFAULT, &Texture, NULL ) );
	return Texture;
}

LPDIRECT3DSURFACE9 EngineCore::CreateDepthStencilSurface(int width, int height)
{
	HRESULT hr;
	LPDIRECT3DSURFACE9 Surface;
	V( device->CreateDepthStencilSurface(   width, height, 
											DXUTGetDeviceSettings().pp.AutoDepthStencilFormat,
                                            D3DMULTISAMPLE_NONE, 0, TRUE, &Surface, NULL ) );
	return Surface;
}

void EngineCore::setGlobalParameters(D3DXMATRIX viewMatrix, D3DXMATRIX projMatrix, D3DXVECTOR3 eyePosition)
{
	effect->SetFloatArray("eyePosition", (float*)(&eyePosition), 3);
	D3DXMATRIX viewProjMatrix = viewMatrix * projMatrix;
	D3DXMATRIX viewProjMatrixInverse;
	D3DXMatrixInverse(&viewProjMatrixInverse, NULL, &viewProjMatrix);
	effect->SetMatrix("viewProjMatrixInverse", &viewProjMatrixInverse);

	SpotLightDirectory::iterator i = spotLightDirectory.begin();
	int ci = 0;
	while(i != spotLightDirectory.end())
	{
		char varname[128];

		sprintf(varname, "spotlights[%d].peakRadiance", ci);
		effect->SetFloatArray(varname,  (float*)&i->second->getPeakRadiance(), 3);
		
		sprintf(varname, "spotlights[%d].position", ci);
		effect->SetFloatArray(varname, (float*)&i->second->getPosition(), 3);
		
		sprintf(varname, "spotlights[%d].direction", ci);
		effect->SetFloatArray(varname, (float*)&i->second->getDirection(), 3);
		
		sprintf(varname, "spotlights[%d].focus", ci);
		effect->SetFloat(varname, i->second->getFocus());
		
		i++; ci++;
	}
}

void EngineCore::render()
{
	if(worldRoot == NULL)
		return;
	 HRESULT hr;

    // Clear the render target and the zbuffer 
    V( device->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(0, 255, 255, 255), 1.0f, 0) );

	device->BeginScene();

	setGlobalParameters(*(camera.GetViewMatrix()), *(camera.GetProjMatrix()), *(camera.GetEyePt()));

	//render world
	D3DXMATRIX rootNodeTransform;
	D3DXMatrixIdentity(&rootNodeTransform);
	worldRoot->render(RenderContext(device, effect, camera.GetProjMatrix(), camera.GetViewMatrix(), &rootNodeTransform, L"default"));

	renderHUD();

	device->EndScene();
}

void EngineCore::renderHUD()
{
	CDXUTTextHelper txtHelper( Font, NULL, 12 );

	txtHelper.Begin();
    txtHelper.SetInsertionPos( 5, 5 );
    txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 0.0f, 0.0f, 1.0f ) );
    txtHelper.DrawTextLine( DXUTGetFrameStats( true ) ); 
    txtHelper.DrawTextLine( DXUTGetDeviceStats() );
	txtHelper.SetInsertionPos( 5, 45 );
	txtHelper.DrawTextLine( L"[WASD] camera movement" );
	txtHelper.DrawTextLine( L"[ZHFGJK] refractor movement" );
	txtHelper.SetInsertionPos( 5, 85 );

	EntityDirectory::iterator i = entityDirectory.begin();
	while( i != entityDirectory.end() )
	{
		EnvironmentMappedEntity* envEntity = dynamic_cast<EnvironmentMappedEntity*>(i->second);	
		if( envEntity != NULL )
		{
			txtHelper.DrawFormattedTextLine( L"[YX] fresnel of the water: %f\n", envEntity->getFresnelFactorWater() );
			txtHelper.DrawFormattedTextLine( L"[CV] fresnel of the refractor: %f\n", envEntity->getFresnelFactorRefractor() );
			txtHelper.DrawFormattedTextLine( L"[LO] fresnel of the container: %f\n", envEntity->getFresnelFactorGlass() );
			txtHelper.DrawFormattedTextLine( L"[BN] IOR of refractor: %f\n", envEntity->getIndexOfRefractionRefractor() );
			
			if( envEntity->areForcesDisabled() ) txtHelper.DrawTextLine( L"[M] disabled forces: true" );
			else txtHelper.DrawTextLine( L"[M] disabled forces: false" );
		}

		i++;
	}
}

Entity* EngineCore::getFullScreenQuad()
{
	Entity* returnValue = NULL;

	EntityDirectory::iterator iEntity = entityDirectory.find(L"fullScreenQuadEntity");
	if(iEntity != entityDirectory.end())
	{
		returnValue = iEntity->second;
	}

	return returnValue;
}

Entity* EngineCore::getEntity(wchar_t* entityName)
{
	Entity* returnValue = NULL;

	EntityDirectory::iterator iEntity = entityDirectory.find(entityName);
	if(iEntity != entityDirectory.end())
	{
		returnValue = iEntity->second;
	}

	return returnValue;
}

SpotLight* EngineCore::getSpotLight(wchar_t* spotLightName)
{
	SpotLight* returnValue = NULL;

	SpotLightDirectory::iterator iSpotLight = spotLightDirectory.find(spotLightName);
	if(iSpotLight != spotLightDirectory.end())
	{
		returnValue = iSpotLight->second;
	}

	return returnValue;
}

void EngineCore::processMessage(HWND hWnd,  UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	worldRoot->handleMessage(hWnd, uMsg, wParam, lParam);
	camera.HandleMessages(hWnd, uMsg, wParam, lParam);
}

void EngineCore::animate(double dt, double t)
{	
	dt*=4;
	camera.FrameMove(dt);
	// animate world
	worldRoot->control(dt, worldRoot);
	worldRoot->animate(dt);
}

void EngineCore::loadLevel(wchar_t* xmlFileName)
{
	//xml root node
	XMLNode xMainNode=XMLNode::openFileHelper(xmlFileName, L"level");

	loadRigidModels(xMainNode);

	//mesh nodes
	int iMesh = 0;
	XMLNode meshNode;
	while( !(meshNode = xMainNode.getChildNode(L"mesh", iMesh)).isEmpty() )
	{
		LPD3DXBUFFER materialBuffer;
		LPD3DXMESH mesh;
		int nSubmeshes;

		//loading the .x file
		const wchar_t* xFileName = meshNode|L"xFileName";
		D3DXLoadMeshFromX(xFileName, D3DXMESH_MANAGED, device, NULL, &materialBuffer, NULL, (DWORD*)&nSubmeshes, &mesh);
		//the name of the mesh
		const wchar_t* meshName = meshNode|L"name";
		//we store the mesh
		meshDirectory[meshName] = mesh;

		if( 
			(meshNode|L"autoShadedMesh") == NULL || wcscmp(meshNode|L"autoShadedMesh", L"false") != 0)
			shadedMeshDirectory[meshName] = new ShadedMesh(mesh, materialBuffer, nSubmeshes, textureDirectory, device);
		iMesh++;
	}

	int iShadedMesh = 0;
	XMLNode shadedMeshNode;
	while( !(shadedMeshNode = xMainNode.getChildNode(L"shadedMesh", iShadedMesh)).isEmpty() )
	{
		//name of the mesh
		const wchar_t* meshName = shadedMeshNode|L"mesh";
		MeshDirectory::iterator iMesh = meshDirectory.find(meshName);
		if(iMesh != meshDirectory.end())
		{
			const wchar_t* shadedMeshName = shadedMeshNode|L"name";		//name of the shadedmesh
			
			ShadedMesh* shadedMesh = NULL;

			if( (shadedMeshNode|L"refractor") == NULL || wcscmp(shadedMeshNode|L"refractor", L"true") != 0 )
				shadedMesh = new ShadedMesh(iMesh->second);
			else 
				shadedMesh = new RefractorMesh(iMesh->second);
			
			int iRole = 0;
			XMLNode roleNode;
			//loading roles
			while( !(roleNode = shadedMeshNode.getChildNode(L"role", iRole)).isEmpty() )
			{
				Role* role = new Role( );

				int iMaterial = 0;
				XMLNode materialNode;
				//loading materials
				while( !(materialNode = roleNode.getChildNode(L"material", iMaterial)).isEmpty() )
				{
					//effect technique
					Material* material = new Material(materialNode||L"technique");

					int iTexture = 0;
					XMLNode textureNode;
					//loading textures
					while( !(textureNode = materialNode.getChildNode(L"texture", iTexture)).isEmpty() )
					{
						
						LPDIRECT3DTEXTURE9 texture;
						const char* textureFileName = textureNode||L"file";
						TextureDirectory::iterator iTex = textureDirectory.find(textureFileName);
						
						if(iTex == textureDirectory.end())
						{
							char textureFilePath[512];
							strcpy(textureFilePath, "Media\\");
							strcat(textureFilePath, textureFileName);
							HRESULT hr = D3DXCreateTextureFromFileA(device, textureFilePath, &texture);
							if(hr != S_OK)
								texture = NULL;
							textureDirectory[textureFileName] = texture;
						}
						else
							texture = iTex->second;
						material->setTexture(textureNode||L"name", texture);
						iTexture++;
					}

					int iCubeTexture = 0;
					XMLNode cubeTextureNode;
					//loading cube maps
					while( !(cubeTextureNode = materialNode.getChildNode(L"cubeTexture", iCubeTexture)).isEmpty() )
					{
						LPDIRECT3DCUBETEXTURE9 cubeTexture;
						const char* textureFileName = cubeTextureNode||L"file";
						CubeTextureDirectory::iterator iTex = cubeTextureDirectory.find(textureFileName);
						
						if(iTex == cubeTextureDirectory.end())
						{
							char textureFilePath[512];
							strcpy(textureFilePath, "Media\\");
							strcat(textureFilePath, textureFileName);
							HRESULT hr = D3DXCreateCubeTextureFromFileA(device, textureFilePath, &cubeTexture);
							if(hr != S_OK)
								cubeTexture = NULL;
							cubeTextureDirectory[textureFileName] = cubeTexture;
						}
						else
							cubeTexture = iTex->second;
						material->setCubeTexture(cubeTextureNode||L"name", cubeTexture);
						iCubeTexture++;
					}

					int iVector = 0;
					XMLNode vectorNode;
					
					while( !(vectorNode = materialNode.getChildNode(L"vector", iVector)).isEmpty() )
					{
						wchar_t* endp;
						const wchar_t* p0String = vectorNode|L"v0";
						double p0 = p0String?wcstod( p0String, &endp):0.0;
						const wchar_t* p1String = vectorNode|L"v1";
						double p1 = p1String?wcstod( p1String, &endp):0.0;
						const wchar_t* p2String = vectorNode|L"v2";
						double p2 = p2String?wcstod( p2String, &endp):0.0;
						const wchar_t* p3String = vectorNode|L"v3";
						double p3 = p3String?wcstod( p3String, &endp):0.0;
						D3DXVECTOR4 value(p0, p1, p2, p3);

						material->setVector(vectorNode||L"name", value);
						iVector++;
					}

					role->addMaterial(material);
					iMaterial++;
				}

				shadedMesh->addRole(roleNode|L"name", role);
				iRole++;
			}

			shadedMeshDirectory[shadedMeshName] = shadedMesh;
			iShadedMesh++;
		}
	}

	XMLNode groupNode = xMainNode.getChildNode(L"group");

	worldRoot = NULL;
	loadSpotlights(xMainNode);
	loadGroup(groupNode, worldRoot);
}

void EngineCore::loadRigidModels(XMLNode& xMainNode)
{
	int iRigidModel = 0;
	XMLNode rigidModelNode;
	while(!(rigidModelNode = xMainNode.getChildNode(L"RigidModel", iRigidModel)).isEmpty())
	{
		const wchar_t* rigidModelName = rigidModelNode|L"name";

		wchar_t* endp;
		const wchar_t* invMassString = rigidModelNode|L"invMass";
		double invMass = invMassString?wcstod( invMassString, &endp):0.0;
		const wchar_t* invAngularMassXString = rigidModelNode|L"invAngularMassX";
		double invAngularMassX = invAngularMassXString?wcstod( invAngularMassXString, &endp):0.0;
		const wchar_t* invAngularMassYString = rigidModelNode|L"invAngularMassY";
		double invAngularMassY = invAngularMassYString?wcstod( invAngularMassYString, &endp):0.0;
		const wchar_t* invAngularMassZString = rigidModelNode|L"invAngularMassZ";
		double invAngularMassZ = invAngularMassZString?wcstod( invAngularMassZString, &endp):0.0;

		const wchar_t* massXString = rigidModelNode|L"centreOfMassX";
		double massX = massXString?wcstod( massXString, &endp):0.0;
		const wchar_t* massYString = rigidModelNode|L"centreOfMassY";
		double massY = massYString?wcstod( massYString, &endp):0.0;
		const wchar_t* massZString = rigidModelNode|L"centreOfMassZ";
		double massZ = massZString?wcstod( massZString, &endp):0.0;
		
		RigidModel* rigidModel = new RigidModel(invMass, D3DXVECTOR3(massX, massY, massZ),
											    invAngularMassX, invAngularMassY, invAngularMassZ);

		rigidModelDirectory[rigidModelName] = rigidModel;
		iRigidModel++;
	}
}

void EngineCore::loadRigidBodies(XMLNode& groupNode, NodeGroup* group)
{
	int iRigidBody = 0;
	XMLNode rigidBodyNode;
	while( !(rigidBodyNode = groupNode.getChildNode(L"RigidBody", iRigidBody)).isEmpty() )
	{
		const wchar_t* shadedMeshName = rigidBodyNode|L"shadedMesh";
		ShadedMeshDirectory::iterator iShadedMesh = shadedMeshDirectory.find(shadedMeshName);
		const wchar_t* rigidModelName = rigidBodyNode|L"rigidModel";
		RigidModelDirectory::iterator iRigidModel = rigidModelDirectory.find(rigidModelName);
		
		if( iShadedMesh != shadedMeshDirectory.end() &&
		    iRigidModel != rigidModelDirectory.end() )
		{
			RigidBody* rigidBody = new RigidBody(iShadedMesh->second, iRigidModel->second);
			
			wchar_t* endp;
			const wchar_t* p0String = rigidBodyNode|L"p0";
			double p0 = p0String?wcstod( p0String, &endp):0.0;
			const wchar_t* p1String = rigidBodyNode|L"p1";
			double p1 = p1String?wcstod( p1String, &endp):0.0;
			const wchar_t* p2String = rigidBodyNode|L"p2";
			double p2 = p2String?wcstod( p2String, &endp):0.0;
			D3DXVECTOR3 position(p0, p1, p2);
			
			rigidBody->setPosition(position);
			group->add(rigidBody);
			const wchar_t* entityName = rigidBodyNode|L"name";
			if(entityName) entityDirectory[entityName] = rigidBody;
		}

		iRigidBody++;
	}
}

void EngineCore::loadGroup(XMLNode& groupNode, NodeGroup*& group)
{
	if(groupNode.isEmpty())
		return;

	group = new NodeGroup();

	loadRigidBodies(groupNode, group);

	int iEntity = 0;
	XMLNode entityNode;
	while( !(entityNode = groupNode.getChildNode(L"entity", iEntity)).isEmpty() )
	{
		const wchar_t* shadedMeshName = entityNode|L"shadedMesh";
		const wchar_t* entityName = entityNode|L"name";
		ShadedMeshDirectory::iterator iShadedMesh = shadedMeshDirectory.find(shadedMeshName);
		if(iShadedMesh != shadedMeshDirectory.end())
		{
			Entity* entity = new Entity(iShadedMesh->second);

			wchar_t* endp;
			const wchar_t* p0String = entityNode|L"p0";
			double p0 = p0String?wcstod( p0String, &endp):0.0;
			const wchar_t* p1String = entityNode|L"p1";
			double p1 = p1String?wcstod( p1String, &endp):0.0;
			const wchar_t* p2String = entityNode|L"p2";
			double p2 = p2String?wcstod( p2String, &endp):0.0;
			D3DXVECTOR3 position(p0, p1, p2);

			entity->setPosition(position);

			entityDirectory[entityName] = entity;

			group->add(entity);
		}
		iEntity++;
	}
	
	

	int iEnvMapEntity = 0;
	XMLNode envMapEntityNode;
	while( !(envMapEntityNode = groupNode.getChildNode(L"environmentMappedEntity", iEnvMapEntity)).isEmpty() )
	{
		const wchar_t* shadedMeshName = envMapEntityNode|L"shadedMesh";
		const wchar_t* entityName = envMapEntityNode|L"name";
		ShadedMeshDirectory::iterator iShadedMesh = shadedMeshDirectory.find(shadedMeshName);
		if(iShadedMesh != shadedMeshDirectory.end())
		{
			EnvironmentMappedEntity* envMapEntity = new EnvironmentMappedEntity(iShadedMesh->second);

			wchar_t* endp;
			const wchar_t* xString = envMapEntityNode|L"p0";
			double p0 = xString?wcstod( xString, &endp):0.0;
			const wchar_t* yString = envMapEntityNode|L"p1";
			double p1 = yString?wcstod( yString, &endp):0.0;
			const wchar_t* zString = envMapEntityNode|L"p2";
			double p2 = zString?wcstod( zString, &endp):0.0;
			D3DXVECTOR3 position(p0, p1, p2);
			envMapEntity->setPosition(position);

			xString = envMapEntityNode|L"envMapPos.x";
			double envx = xString?wcstod( xString, &endp):0.0;
			yString = envMapEntityNode|L"envMapPos.y";
			double envy = yString?wcstod( yString, &endp):0.0;
			zString = envMapEntityNode|L"envMapPos.z";
			double envz = zString?wcstod( zString, &endp):0.0;
			D3DXVECTOR3 envMapPosition(envx, envy, envz);
			envMapEntity->setEnvironmentMapPosition(envMapPosition);

			
			XMLNode refractorNode;
			if( !(refractorNode = envMapEntityNode.getChildNode(L"refractor", 0)).isEmpty() )
			{
				const wchar_t* refractorName = refractorNode|L"name";
				EntityDirectory::iterator iRefractorEntity = entityDirectory.find(refractorName);
				if(iRefractorEntity != entityDirectory.end())
				{	
					envMapEntity->addRefractor(iRefractorEntity->second);
				}
			}

			XMLNode lightNode;
			if( !(lightNode = envMapEntityNode.getChildNode(L"causticLight", 0)).isEmpty() )
			{
				const wchar_t* lightName = lightNode|L"name";
				SpotLightDirectory::iterator iSpotLight = spotLightDirectory.find(lightName);
				if(iSpotLight != spotLightDirectory.end())
				{	
					envMapEntity->addCausticLight(iSpotLight->second);
				}
			}
			
			entityDirectory[entityName] = envMapEntity;

			group->add(envMapEntity);
		}
		iEnvMapEntity++;
	}


	int iSubGroup = 0;
	XMLNode subGroupNode;
	while( !(subGroupNode = groupNode.getChildNode(L"group", iSubGroup)).isEmpty() )
	{
		NodeGroup* subGroup = NULL;
		loadGroup(subGroupNode, subGroup);

		if(subGroup)
			group->add(subGroup);
		iSubGroup++;
	}

}

void EngineCore::loadSpotlights(XMLNode& xMainNode)
{
	int iSpotlight = 0;
	XMLNode spotlightNode;

	while( !(spotlightNode = xMainNode.getChildNode(L"Spotlight", iSpotlight)).isEmpty() )
	{
		

			wchar_t* endp;
			const wchar_t* xString = spotlightNode|L"pos.x";
			double x = xString?wcstod( xString, &endp):0.0;
			const wchar_t* yString = spotlightNode|L"pos.y";
			double y = yString?wcstod( yString, &endp):0.0;
			const wchar_t* zString = spotlightNode|L"pos.z";
			double z = zString?wcstod( zString, &endp):0.0;
			D3DXVECTOR3 position = D3DXVECTOR3(x, y, z);

			xString = spotlightNode|L"dir.x";
			double x2 = xString?wcstod( xString, &endp):0.0;
			yString = spotlightNode|L"dir.y";
			double y2 = yString?wcstod( yString, &endp):0.0;
			zString = spotlightNode|L"dir.z";
			double z2 = zString?wcstod( zString, &endp):0.0;
			D3DXVECTOR3 direction = D3DXVECTOR3(x2, y2, z2);

			xString = spotlightNode|L"power.r";
			double x3 = xString?wcstod( xString, &endp):0.0;
			yString = spotlightNode|L"power.g";
			double y3 = yString?wcstod( yString, &endp):0.0;
			zString = spotlightNode|L"power.b";
			double z3 = zString?wcstod( zString, &endp):0.0;
			D3DXVECTOR3 peakRadiance = D3DXVECTOR3(x3, y3, z3);
		
			xString = spotlightNode|L"focus";
			x = xString?wcstod( xString, &endp):0.0;
			double focus = x;

		SpotLight* spotlight = new SpotLight(peakRadiance, position, direction, focus);
		const wchar_t* ownerEntityName = spotlightNode|L"owner";
		if(ownerEntityName != NULL)
		{
			EntityDirectory::iterator iEntity = entityDirectory.find(ownerEntityName);
			if(iEntity != entityDirectory.end())
			{
				spotlight->setOwner(iEntity->second);
			}
		}

		const wchar_t* spotlightName = spotlightNode|L"name";
		spotLightDirectory[spotlightName] = spotlight;
		iSpotlight++;
	}
}

