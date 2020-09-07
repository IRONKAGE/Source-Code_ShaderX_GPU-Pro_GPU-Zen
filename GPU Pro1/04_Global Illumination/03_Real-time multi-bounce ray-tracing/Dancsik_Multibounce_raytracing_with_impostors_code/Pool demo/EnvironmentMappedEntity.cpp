#include "dxstdafx.h"
#include "EnvironmentMappedEntity.h"
#include "ShadedMesh.h"
#include "NodeGroup.h"
#include "EngineCore.h"
#include "RigidBody.h"
#include "RefractorMesh.h"
#include "SpotLight.h"

EnvironmentMappedEntity::EnvironmentMappedEntity(ShadedMesh* shadedMesh) : Entity(shadedMesh)
{
	core = NULL;

	disableForces = false;
	updateFresnelWater = true;
	updateFresnelRefractor = true;
	updateFresnelGlass = true;
	updateRefractorPosition = false;
	updateRefractorIOR = true;
	indexOfRefractionRefractor = 1.0f;
	time = 0;
	fresnelFactorWater = 0.02f;
	fresnelFactorRefractor = 0.04f;
	fresnelFactorGlass = 0.14f;
	refractor = NULL;
	fullScreenQuad = NULL;
	photonMapSize = 256;
	snippetSize = 0.011;
}

EnvironmentMappedEntity::~EnvironmentMappedEntity(void)
{
}

HRESULT EnvironmentMappedEntity::createDefaultResources(EngineCore* core)
{
	this->core = core;

	shadedMesh->createDefaultResources(core);

	fullScreenQuad = core->getFullScreenQuad();
	

	uvMap = core->CreateCubeTexture(512, D3DFMT_A16B16G16R16F);
	normalMapGlass = core->CreateCubeTexture(512, D3DFMT_A16B16G16R16F);
	environmentSurface = core->CreateDepthStencilSurface(512, 512);
	
	RenderSceneIntoCubeMap(RENDER_ENVIRONMENT_UV_DISTANCE);
	RenderSceneIntoCubeMap(RENDER_GLASS_NORMAL_DISTANCE);

	core->getEffect()->SetTexture("normalMapTexture", normalMapGlass);
	core->getEffect()->SetTexture("uvMapTexture", uvMap);
	

	heightMapSurface = core->CreateDepthStencilSurface(128, 128);
	heightMap = core->CreateTexture(128, 128, D3DFMT_A16B16G16R16F);
	

	photonMapSurface = core->CreateDepthStencilSurface(photonMapSize, photonMapSize);
	photonMap = core->CreateTexture(photonMapSize, photonMapSize, D3DFMT_A16B16G16R16F);
	lightMap = core->CreateTexture(512, 512, D3DFMT_A16B16G16R16F);
	lightMapStencilSurface = core->CreateDepthStencilSurface(512, 512);
	D3DXCreateTextureFromFile( core->getDevice(), L"Media\\PowerOfSnippetTexel.dds", &gaussianFilterTexture);
	core->getEffect()->SetTexture("gaussianFilterTexture", gaussianFilterTexture);

	//*******************************************

	//we calculate the bounding box of the water surface

	D3DXVECTOR4 amplitudes = fullScreenQuad->getMesh()->getRole(L"createWaterHeightMap")->getMaterial(0)->getVector(L"Amplitudes");
	float maximumAmplitude = amplitudes.x + amplitudes.y + amplitudes.z + amplitudes.w;

	if( maximumAmplitude == 0 )
	{
		fullScreenQuad->getMesh()->getRole(L"createWaterHeightMap")->getMaterial(0)->setVector(L"Amplitudes", D3DXVECTOR4(0.01, 0, 0, 0));
		maximumAmplitude = 0.01;
	}

	float bBoxScale = 1.3f;	//we will enlarge the bounding box

	
	float heightMapScale = 0.5f/(bBoxScale*maximumAmplitude);
	float heightMapOffset = 0.5f - 0.5f/bBoxScale;

	shadedMesh->boundingBoxMin.y -= bBoxScale * maximumAmplitude;
	shadedMesh->boundingBoxMax.y += bBoxScale * maximumAmplitude;

	D3DXVECTOR3 bmin  = shadedMesh->boundingBoxMin;
	D3DXVECTOR3 bmax = shadedMesh->boundingBoxMax;
	
	//*******************************************

	//We will calculate the intersection with the water surface's height map in the following coordinate system:
	//the centre will be in 'bmin' and the three unit long axes will be the three edge of the bounding box at 'bmin'.

	//The matrices 'TBNInverse' and 'TBN' will change the world space to the water surface's coordinate system
	//and the water surface's coordinate system to world space.

	D3DXMatrixIdentity(&TBN);
	D3DXMatrixTranslation(&TBN, bmin.x + position.x, bmin.y + position.y, bmin.z + position.z);
	TBN._11 = bmax.x - bmin.x;
	TBN._22 = 2*bBoxScale*maximumAmplitude;
	TBN._33 = bmax.z - bmin.z;
	core->getEffect()->SetMatrix("TBN", &TBN);

	
	D3DXMATRIXA16 translationMatrix;
	D3DXMatrixIdentity(&translationMatrix);
	D3DXMatrixTranslation(&translationMatrix, -TBN._41, -TBN._42, -TBN._43);
	D3DXMATRIXA16 TBNInverse = D3DXMATRIXA16(TBN);
	float lengthSq1 = D3DXVec3LengthSq( &D3DXVECTOR3(TBNInverse._11, TBNInverse._12, TBNInverse._13) );
	float lengthSq2 = D3DXVec3LengthSq( &D3DXVECTOR3(TBNInverse._21, TBNInverse._22, TBNInverse._23) );
	float lengthSq3 = D3DXVec3LengthSq( &D3DXVECTOR3(TBNInverse._31, TBNInverse._32, TBNInverse._33) );
	TBNInverse._11 /= lengthSq1; TBNInverse._21 /= lengthSq1; TBNInverse._31 /= lengthSq1;
	TBNInverse._12 /= lengthSq2; TBNInverse._22 /= lengthSq2; TBNInverse._32 /= lengthSq2;
	TBNInverse._13 /= lengthSq3; TBNInverse._23 /= lengthSq3; TBNInverse._33 /= lengthSq3;
	TBNInverse._41 = 0; TBNInverse._42 = 0; TBNInverse._43 = 0;
	D3DXMatrixMultiply(&TBNInverse, &translationMatrix, &TBNInverse);
	core->getEffect()->SetMatrix("TBNInverse", &TBNInverse);

	//*******************************************
	
	fullScreenQuad->getMesh()->getRole(L"createWaterHeightMap")->getMaterial(0)->setVector("boundingBoxMin", D3DXVECTOR4(bmin.x, bmin.y, bmin.z, 1));
	fullScreenQuad->getMesh()->getRole(L"createWaterHeightMap")->getMaterial(0)->setVector("boundingBoxMax", D3DXVECTOR4(bmax.x, bmax.y, bmax.z, 1));
	
	core->getEffect()->SetVector("maximumAmplitude", &D3DXVECTOR4(maximumAmplitude, 0, 0, 0));
	core->getEffect()->SetVector("heightMapScale", &D3DXVECTOR4(heightMapScale, 0, 0, 0));
	core->getEffect()->SetVector("heightMapOffset", &D3DXVECTOR4(heightMapOffset, 0, 0, 0));
	core->getEffect()->CommitChanges();
	//*******************************************
	//we load the textures for the water shader:
	D3DXVECTOR3 referencePoint = getReferencePointPosition();
	shadedMesh->getRole(L"default")->getMaterial(0)->setVector("referencePointPosition", D3DXVECTOR4(referencePoint.x, referencePoint.y, referencePoint.z, 1));
	bmin = refractor->getMesh()->boundingBoxMin;
	bmax = refractor->getMesh()->boundingBoxMax;
	shadedMesh->getRole(L"default")->getMaterial(0)->setVector("boundingBoxMin", D3DXVECTOR4(bmin.x, bmin.y, bmin.z, 1));
	shadedMesh->getRole(L"default")->getMaterial(0)->setVector("boundingBoxMax", D3DXVECTOR4(bmax.x, bmax.y, bmax.z, 1));
	shadedMesh->getRole(L"default")->getMaterial(0)->setCubeTexture("refractorMapTexture",  ((RefractorMesh*)refractor->getMesh())->getRefractorMap());
	
	shadedMesh->getRole(L"createPhotonMap")->getMaterial(0)->setVector("boundingBoxMin", D3DXVECTOR4(bmin.x, bmin.y, bmin.z, 1));
	shadedMesh->getRole(L"createPhotonMap")->getMaterial(0)->setVector("boundingBoxMax", D3DXVECTOR4(bmax.x, bmax.y, bmax.z, 1));
	shadedMesh->getRole(L"createPhotonMap")->getMaterial(0)->setCubeTexture("refractorMapTexture",  ((RefractorMesh*)refractor->getMesh())->getRefractorMap());
	shadedMesh->getRole(L"createPhotonMap")->getMaterial(0)->setVector("referencePointPosition", D3DXVECTOR4(referencePoint.x, referencePoint.y, referencePoint.z, 1));
	shadedMesh->getRole(L"createPhotonMap")->getMaterial(0)->setCubeTexture("uvMapTexture", uvMap);

	
	//*******************************************
	//the textures for the floating object's shader
	refractor->getMesh()->getRole(L"default")->getMaterial(0)->setVector("referencePointPosition", D3DXVECTOR4(referencePoint.x, referencePoint.y, referencePoint.z, 1));
	refractor->getMesh()->getRole(L"default")->getMaterial(0)->setCubeTexture("refractorMapTexture",  ((RefractorMesh*)refractor->getMesh())->getRefractorMap());
	refractor->getMesh()->getRole(L"createPhotonMap")->getMaterial(0)->setVector("referencePointPosition", D3DXVECTOR4(referencePoint.x, referencePoint.y, referencePoint.z, 1));
	refractor->getMesh()->getRole(L"createPhotonMap")->getMaterial(0)->setCubeTexture("refractorMapTexture",  ((RefractorMesh*)refractor->getMesh())->getRefractorMap());
	
	D3DXMATRIX  sphereModelMatrix, sphereModelMatrixInverse;
	((RigidBody*)refractor)->getModelMatrix(sphereModelMatrix);
	((RigidBody*)refractor)->getModelMatrixInverse(sphereModelMatrixInverse);
	D3DXMatrixInverse(&sphereModelMatrixInverse, NULL, &sphereModelMatrix);

	core->getEffect()->SetMatrix("sphereModelMatrix", &sphereModelMatrix);
	core->getEffect()->SetMatrix("sphereModelMatrixInverse", &sphereModelMatrixInverse);

	createCausticQuadrilaterals();

	return S_OK;
}

void EnvironmentMappedEntity::createCausticQuadrilaterals()
{
	D3DVERTEXELEMENT9 causticQuadVertexElements[] =
	{	//the xy values store the quad size and the zw values store the texture coordinates of the quad.
		//These attributes are identical for every quad.
		{0, 0, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
		//The xy values store the position of the quad in texture space which are unique for every quad. 
		{1, 0, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, 	D3DDECLUSAGE_POSITION, 0},
		D3DDECL_END() 
	};

	core->getDevice()->CreateVertexDeclaration(causticQuadVertexElements, &causticQuadVertexDecl);


	core->getDevice()->CreateVertexBuffer( 4 * sizeof(D3DXVECTOR4),
										   D3DUSAGE_DYNAMIC|D3DUSAGE_WRITEONLY,
										   D3DFVF_XYZW,
										   D3DPOOL_DEFAULT,
										   &causticQuadVertexBuffer,
										   NULL
										  );

	D3DXVECTOR4* vertexData;
	causticQuadVertexBuffer->Lock(0, 0, (void**)&vertexData, D3DLOCK_DISCARD);
	
	//float4(offset.x, offset.y, texcoord.u, texcoord.v)
	vertexData[0] = D3DXVECTOR4(-snippetSize, -snippetSize, 0.0f, 0.0f);
	vertexData[1] = D3DXVECTOR4(snippetSize, -snippetSize, 1.0f, 0.0f);
	vertexData[2] = D3DXVECTOR4(snippetSize, snippetSize, 1.0f, 1.0f);
	vertexData[3] = D3DXVECTOR4(-snippetSize, snippetSize, 0.0f, 1.0f);	

	causticQuadVertexBuffer->Unlock();

	core->getDevice()->CreateIndexBuffer(6 * sizeof(short), 0, D3DFMT_INDEX16, D3DPOOL_DEFAULT, &causticQuadIndexBuffer, NULL);

	short* indexData;
	causticQuadIndexBuffer->Lock(0, 0, (void**)&indexData, D3DLOCK_DISCARD);
	indexData[0] = 0; indexData[1] = 3; indexData[2] = 1; indexData[3] = 2; indexData[4] = 1; indexData[5] = 3;
	causticQuadIndexBuffer->Unlock();

	
	//we create and fill the instance buffer
	core->getDevice()->CreateVertexBuffer(	2 * photonMapSize * photonMapSize * sizeof(float),
											D3DUSAGE_DYNAMIC, 0,
											D3DPOOL_DEFAULT,
											&causticQuadInstanceBuffer,
											NULL
										  );

	
	float* instanceData;
	causticQuadInstanceBuffer->Lock(0, 0, (void**)&instanceData, D3DLOCK_DISCARD);

	int counter = 0;
	for ( int i = 0; i < photonMapSize; i++ )
	{
		for ( int j = 0; j < photonMapSize; j++ )
		{
			float u = (float) i / photonMapSize;
			float v = (float) j / photonMapSize;
			
			instanceData[counter] = u; counter += 1;
			instanceData[counter] = v; counter += 1;
		}
	}
	causticQuadInstanceBuffer->Unlock();
}

HRESULT EnvironmentMappedEntity::releaseDefaultResources()
{
	shadedMesh->releaseDefaultResources();
	uvMap->Release();
	normalMapGlass->Release();
	environmentSurface->Release();
	heightMap->Release();
	heightMapSurface->Release();
	photonMapSurface->Release();
	photonMap->Release();
	lightMap->Release();
	lightMapStencilSurface->Release();
	causticQuadVertexDecl->Release();
	causticQuadVertexBuffer->Release();
	causticQuadIndexBuffer->Release();
	causticQuadInstanceBuffer->Release();
	gaussianFilterTexture->Release();

	return S_OK;
}

void EnvironmentMappedEntity::setEnvironmentMapPosition(const D3DXVECTOR3& position)
{
	this->environmentMapPosition = position;
}

void EnvironmentMappedEntity::RenderSceneIntoCubeMap(int mode)
{
	if( !( mode==RENDER_ENVIRONMENT_UV_DISTANCE || mode==RENDER_GLASS_NORMAL_DISTANCE ) ) return;
   
	LPDIRECT3DSURFACE9 oldTarget = NULL;
	core->getDevice()->GetRenderTarget( 0, &oldTarget );
    LPDIRECT3DSURFACE9 oldStencil = NULL;
    if( SUCCEEDED( core->getDevice()->GetDepthStencilSurface( &oldStencil ) ) )
    { 
		core->getDevice()->SetDepthStencilSurface( environmentSurface );
    }

	D3DXMATRIXA16 mProj;
	D3DXMatrixPerspectiveFovLH( &mProj, D3DX_PI * 0.5f, 1.0f, 0.001f, 3000.0f );

    for(DWORD i=0; i<6; i++)
    {
        
		D3DXVECTOR3 vEnvEyePt = getReferencePointPosition();

		D3DXVECTOR3 vLookatPt, vUpVec, vLookat;

        switch(i)
        {
            case D3DCUBEMAP_FACE_POSITIVE_X:
				vLookat = D3DXVECTOR3(1.0f, 0.0f, 0.0f);
                vUpVec    = D3DXVECTOR3(0.0f, 1.0f, 0.0f);
                break;
            case D3DCUBEMAP_FACE_NEGATIVE_X:
				vLookat = D3DXVECTOR3(-1.0f, 0.0f, 0.0f);
				vUpVec    = D3DXVECTOR3( 0.0f, 1.0f, 0.0f);
                break;
            case D3DCUBEMAP_FACE_POSITIVE_Y:
				vLookat = D3DXVECTOR3(0.0f, 1.0f, 0.0f);
                vUpVec    = D3DXVECTOR3(0.0f, 0.0f,-1.0f);
                break;
            case D3DCUBEMAP_FACE_NEGATIVE_Y:
				vLookat = D3DXVECTOR3(0.0f,-1.0f, 0.0f);
                vUpVec    = D3DXVECTOR3(0.0f, 0.0f, 1.0f);
                break;
            case D3DCUBEMAP_FACE_POSITIVE_Z:
				vLookat = D3DXVECTOR3( 0.0f, 0.0f, 1.0f);
                vUpVec    = D3DXVECTOR3( 0.0f, 1.0f, 0.0f);
                break;
            case D3DCUBEMAP_FACE_NEGATIVE_Z:
				vLookat = D3DXVECTOR3(0.0f, 0.0f,-1.0f);
                vUpVec    = D3DXVECTOR3(0.0f, 1.0f, 0.0f);
                break;
        }
		
		vLookatPt = vEnvEyePt + vLookat;
		
        LPDIRECT3DSURFACE9 pFace;
		
		if( mode == RENDER_ENVIRONMENT_UV_DISTANCE )
			uvMap->GetCubeMapSurface((D3DCUBEMAP_FACES)i, 0, &pFace);
		else if( mode == RENDER_GLASS_NORMAL_DISTANCE )
			normalMapGlass->GetCubeMapSurface((D3DCUBEMAP_FACES)i, 0, &pFace);

		core->getDevice()->SetRenderTarget (0 , pFace);
		SAFE_RELEASE(pFace);
		
		core->getDevice()->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0);
		
		core->getDevice()->BeginScene();
	
        D3DXMATRIX matView;
		D3DXMatrixLookAtLH(&matView, &vEnvEyePt, &vLookatPt, &vUpVec);
		D3DXMATRIX rootNodeTransform;
		D3DXMatrixIdentity(&rootNodeTransform);
		
		core->setGlobalParameters(matView, mProj, vEnvEyePt);

		if( mode == RENDER_ENVIRONMENT_UV_DISTANCE )
			core->getWorldRoot()->render(RenderContext(core->getDevice(), core->getEffect(), &mProj,  &matView, &rootNodeTransform, L"storeColorAndUVDistance"));
		else if( mode == RENDER_GLASS_NORMAL_DISTANCE )
			core->getWorldRoot()->render(RenderContext(core->getDevice(), core->getEffect(), &mProj,  &matView, &rootNodeTransform, L"storeColorDistance"));
		
		core->getDevice()->EndScene();
	
	}

	if( oldStencil )
    {
        core->getDevice()->SetDepthStencilSurface( oldStencil );
        SAFE_RELEASE( oldStencil );
    }
    core->getDevice()->SetRenderTarget( 0, oldTarget );
    SAFE_RELEASE( oldTarget );
}

void EnvironmentMappedEntity::renderHeightMap()
{
	LPDIRECT3DSURFACE9 oldTarget = NULL;
	core->getDevice()->GetRenderTarget( 0, &oldTarget );
    LPDIRECT3DSURFACE9 oldStencil = NULL;
    if( SUCCEEDED( core->getDevice()->GetDepthStencilSurface( &oldStencil ) ) )
    { 
			core->getDevice()->SetDepthStencilSurface( heightMapSurface );
    }

	
	LPDIRECT3DSURFACE9 surface;
	heightMap->GetSurfaceLevel(0, &surface);
	core->getDevice()->SetRenderTarget(0, surface);

	core->getDevice()->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0);
		
	core->getDevice()->BeginScene();
	
	RenderContext context = RenderContext(core->getDevice(), core->getEffect(), NULL, NULL, NULL, L"createWaterHeightMap");
	
	fullScreenQuad->getMesh()->render(context);

	core->getDevice()->EndScene();

	//D3DXSaveSurfaceToFile( L"HeightMap.jpg", D3DXIFF_JPG,  surface, NULL, NULL);
	surface->Release();
	
	if( oldStencil )
    {
        core->getDevice()->SetDepthStencilSurface( oldStencil );
        SAFE_RELEASE( oldStencil );
    }
    core->getDevice()->SetRenderTarget( 0, oldTarget );
    SAFE_RELEASE( oldTarget );
}

void EnvironmentMappedEntity::renderCaustics()
{
	LPDIRECT3DSURFACE9 oldTarget = NULL;
	core->getDevice()->GetRenderTarget( 0, &oldTarget );
    LPDIRECT3DSURFACE9 oldStencil = NULL;
    if( SUCCEEDED( core->getDevice()->GetDepthStencilSurface( &oldStencil ) ) )
    { 
		core->getDevice()->SetDepthStencilSurface( photonMapSurface );
    }

	//we set the photon map as render target
	LPDIRECT3DSURFACE9 surface;
	photonMap->GetSurfaceLevel(0, &surface);
	core->getDevice()->SetRenderTarget(0, surface);
	
	core->getDevice()->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0);
	

	core->getDevice()->BeginScene();
	

	//we set the view and projection matrix
	D3DXMATRIXA16 viewMatrix;
	D3DXMATRIXA16 projMatrix;

	D3DXVECTOR4 boundingSphereData = computeBoundingSphere(this, refractor);	//we determine the bounding sphere of the water surface and the floating object
	D3DXVECTOR3 centerObjectPos = D3DXVECTOR3(boundingSphereData.x, boundingSphereData.y, boundingSphereData.z);
	float radiusCenter = 1.03f * boundingSphereData.w;

	D3DXVECTOR3 lightPos = causticLight->getPosition(); 
	
	core->getEffect()->SetVector("eyePosition", &D3DXVECTOR4(lightPos.x, lightPos.y, lightPos.z, 1));

	D3DXVECTOR3 up = D3DXVECTOR3(0, 1, 0);
	D3DXMatrixLookAtLH( &viewMatrix, &lightPos, &centerObjectPos, &up );

	D3DXVECTOR3 diff = lightPos - centerObjectPos;					
	float dist = D3DXVec3Length(&diff);
	

	float fov = atan( radiusCenter / dist ) * 2;		// fov/2 = tan(radiuscenter/dist)
		
	D3DXMatrixPerspectiveFovLH( &projMatrix, fov, 1.0f, 0.001f, 1000.0f );


	
	RenderContext context = RenderContext(core->getDevice(), core->getEffect(), &projMatrix, &viewMatrix, NULL, L"createPhotonMap");
	
	//rendering
	refractor->render(context);  
	render(context);

	core->getDevice()->EndScene();

	//D3DXSaveSurfaceToFile( L"photonMap.png", D3DXIFF_PNG,  surface, NULL, NULL);
	surface->Release();
	
	//we render photon hits in the lightMap
	
	//we set thelight map as render target
	lightMap->GetSurfaceLevel(0, &surface);
	core->getDevice()->SetRenderTarget(0, surface);
	core->getDevice()->SetDepthStencilSurface(lightMapStencilSurface);

	core->getDevice()->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0);
	
	core->getEffect()->SetTexture("photonMap", photonMap);

	//rendering

	core->getDevice()->BeginScene();

	UINT a;
	core->getEffect()->SetTechnique("renderPhotonHit");
	core->getEffect()->Begin(&a, 0);
	core->getEffect()->BeginPass(0);


	core->getDevice()->SetStreamSourceFreq(0, D3DSTREAMSOURCE_INDEXEDDATA | photonMapSize*photonMapSize);
	core->getDevice()->SetStreamSource(0, causticQuadVertexBuffer, 0, sizeof(D3DXVECTOR4));
	core->getDevice()->SetStreamSourceFreq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1);
	core->getDevice()->SetStreamSource(1, causticQuadInstanceBuffer, 0, 2*sizeof(float));
	core->getDevice()->SetVertexDeclaration(causticQuadVertexDecl);
	core->getDevice()->SetIndices(causticQuadIndexBuffer);

	core->getDevice()->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2);

	core->getDevice()->SetStreamSourceFreq(0, 1);
	core->getDevice()->SetStreamSource(0, NULL, 0, 0);
	core->getDevice()->SetStreamSourceFreq(1, 1);
	core->getDevice()->SetStreamSource(1, NULL, 0, 0);
	core->getDevice()->SetIndices(0);


	core->getEffect()->EndPass();
	core->getEffect()->End();

	core->getDevice()->EndScene();

	core->getDevice()->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);

	//D3DXSaveSurfaceToFile( L"LightMap.jpg", D3DXIFF_JPG,  surface, NULL, NULL);

	surface->Release();

	core->getEffect()->SetTexture("lightMap", lightMap);

	if( oldStencil )
    {
        core->getDevice()->SetDepthStencilSurface( oldStencil );
        SAFE_RELEASE( oldStencil );
    }
    core->getDevice()->SetRenderTarget( 0, oldTarget );
    SAFE_RELEASE( oldTarget );
}

D3DXVECTOR4 EnvironmentMappedEntity::computeBoundingSphere(Entity* object1, Entity* object2)
{
	D3DXVECTOR3 center1 = 0.5f * ( object1->getMesh()->boundingBoxMax + object1->getMesh()->boundingBoxMin );
	float radius1 = D3DXVec3Length(&(object1->getMesh()->boundingBoxMax - center1));
	
	D3DXVECTOR3 center2 = 0.5f * ( object2->getMesh()->boundingBoxMax + object2->getMesh()->boundingBoxMin );
	float radius2 = D3DXVec3Length(&(object2->getMesh()->boundingBoxMax - center2));
	
	D3DXVECTOR3 center3 = 0.5f * ( center1 + center2 );
	float radius3 = D3DXVec3Length(&(center3 - center1)) + max(radius1, radius2);
	
	return D3DXVECTOR4(center3.x, center3.y, center3.z, radius3);	//returns the bounding sphere of two bounding spheres
}

D3DXVECTOR3 EnvironmentMappedEntity::getReferencePointPosition()
{
	return position + environmentMapPosition;		//returns the reference position in world space
}

float EnvironmentMappedEntity::getWaterHeightAndNormal(D3DXVECTOR3 location, D3DXVECTOR3& normal)
{
	D3DXVECTOR4 amplitudes = fullScreenQuad->getMesh()->getRole(L"createWaterHeightMap")->getMaterial(0)->getVector(L"Amplitudes");
	D3DXVECTOR4 timeFreq = fullScreenQuad->getMesh()->getRole(L"createWaterHeightMap")->getMaterial(0)->getVector(L"TimeFreq");
	D3DXVECTOR4 spaceFreq = fullScreenQuad->getMesh()->getRole(L"createWaterHeightMap")->getMaterial(0)->getVector(L"SpaceFreq");
	D3DXVECTOR4 waveDirX = fullScreenQuad->getMesh()->getRole(L"createWaterHeightMap")->getMaterial(0)->getVector(L"WaveDirX");
	D3DXVECTOR4 waveDirZ = fullScreenQuad->getMesh()->getRole(L"createWaterHeightMap")->getMaterial(0)->getVector(L"WaveDirZ");

	D3DXVECTOR4 phase = (waveDirX * location.x + waveDirZ * location.z);
	phase.x *= spaceFreq.x; phase.y *= spaceFreq.y; phase.z *= spaceFreq.z; phase.w *= spaceFreq.w;
	phase += timeFreq * time;

	D3DXVECTOR4 cosPhase = D3DXVECTOR4(cos(phase.x), cos(phase.y), cos(phase.z), cos(phase.w));
	D3DXVECTOR4 sinPhase = D3DXVECTOR4(sin(phase.x), sin(phase.y), sin(phase.z), sin(phase.w));

	D3DXVECTOR4 cosWaveHeight = D3DXVECTOR4(cosPhase.x * amplitudes.x * spaceFreq.x,
											cosPhase.y * amplitudes.y * spaceFreq.y,
											cosPhase.z * amplitudes.z * spaceFreq.z,
											cosPhase.w * amplitudes.w * spaceFreq.w
										   );

	D3DXVECTOR3 tangent  = D3DXVECTOR3(1.0f, D3DXVec4Dot(&cosWaveHeight, &waveDirX), 0.0f);	
	D3DXVec3Normalize(&tangent, &tangent);

	D3DXVECTOR3 binormal  = D3DXVECTOR3(0.0f, D3DXVec4Dot(&cosWaveHeight, &waveDirZ), 1.0f);
	D3DXVec3Normalize(&binormal, &binormal);

	D3DXVec3Cross(&normal,  &binormal, &tangent);
	D3DXVec3Normalize(&normal, &normal);

	return D3DXVec4Dot(&sinPhase, &amplitudes);
}

void EnvironmentMappedEntity::control(double dt, Node* others)
{
	if( updateRefractorPosition || disableForces )	//the forces are disabled or the user changes the position of the floating object 
	{
		((RigidBody*)refractor)->initMomentums();
	}
	else
	{
		D3DXVECTOR3 spherePosition = refractor->getPosition();
		D3DXVECTOR3 normal;

		//the radius of the floating object
		float radius = max( D3DXVec3Length(&refractor->getMesh()->boundingBoxMin), D3DXVec3Length(&refractor->getMesh()->boundingBoxMax) );
		
		//the height of the water volume and the normal vector of the water surface are calculated
		float waterHeight = getWaterHeightAndNormal(spherePosition, normal);

		D3DXVECTOR3 waterPos = D3DXVECTOR3(0, (shadedMesh->boundingBoxMin.y+shadedMesh->boundingBoxMax.y)/2 + waterHeight - spherePosition.y, 0);

		float d = D3DXVec3Dot(&waterPos, &normal);
		float absd = abs(d);

		D3DXVECTOR3 F = D3DXVECTOR3(0, 1, 0);		//buoyancy
		float VolumeInWater = 0;

		if( absd < radius )		//if just a part of the sphere is underwater...
		{	//...we approximate the volume in the water with a spherical cap
			float h = radius-absd;
			VolumeInWater = D3DX_PI*h*h*(radius-h/3);	//calculating the volume of the spherical cap
			F = D3DXVECTOR3(normal.x, normal.y, normal.z);
		}

		float density = 1.2f;	//the density of air
		if( d > 0 ) {	//if the centre of the bounding sphere is under the water...
			VolumeInWater = 4*D3DX_PI*radius*radius*radius/3 - VolumeInWater;
			density = 1000;		//the density of water
		}

		F *= VolumeInWater*10000;
		((RigidBody*)refractor)->setForces(F, density);
	}
}

void EnvironmentMappedEntity::animate(double dt)
{
	if( updateFresnelWater )
	{
		shadedMesh->getRole(L"default")->getMaterial(0)->setVector("FresnelFactorWater", D3DXVECTOR4(fresnelFactorWater, 0, 0, 0));
		shadedMesh->getRole(L"createPhotonMap")->getMaterial(0)->setVector("FresnelFactorWater", D3DXVECTOR4(fresnelFactorWater, 0, 0, 0));
		refractor->getMesh()->getRole(L"default")->getMaterial(0)->setVector("FresnelFactorWater", D3DXVECTOR4(fresnelFactorWater, 0, 0, 0));
		refractor->getMesh()->getRole(L"createPhotonMap")->getMaterial(0)->setVector("FresnelFactorWater", D3DXVECTOR4(fresnelFactorWater, 0, 0, 0));
		updateFresnelWater = false;
	}

	if( updateFresnelRefractor )
	{
		shadedMesh->getRole(L"default")->getMaterial(0)->setVector("FresnelFactorRefractor", D3DXVECTOR4(fresnelFactorRefractor, 0, 0, 0));
		shadedMesh->getRole(L"createPhotonMap")->getMaterial(0)->setVector("FresnelFactorRefractor", D3DXVECTOR4(fresnelFactorRefractor, 0, 0, 0));
		refractor->getMesh()->getRole(L"default")->getMaterial(0)->setVector("FresnelFactorRefractor", D3DXVECTOR4(fresnelFactorRefractor, 0, 0, 0));
		refractor->getMesh()->getRole(L"createPhotonMap")->getMaterial(0)->setVector("FresnelFactorRefractor", D3DXVECTOR4(fresnelFactorRefractor, 0, 0, 0));
		updateFresnelRefractor = false;
	}

	if( updateRefractorIOR )
	{
		core->getEffect()->SetFloat("indexOfRefractionRefractor", indexOfRefractionRefractor);
		updateRefractorIOR = false;
	}

	if( updateFresnelGlass )
	{
		core->getEffect()->SetFloat("FresnelFactorGlass", fresnelFactorGlass);
		updateFresnelGlass = false;
	}
	core->getEffect()->CommitChanges();
	shadedMesh->getRole(L"default")->getMaterial(0)->setVector("refractorPosition", D3DXVECTOR4(refractor->getPosition().x, refractor->getPosition().y,refractor->getPosition().z, 1));
	shadedMesh->getRole(L"createPhotonMap")->getMaterial(0)->setVector("refractorPosition", D3DXVECTOR4(refractor->getPosition().x, refractor->getPosition().y,refractor->getPosition().z, 1));
	refractor->getMesh()->getRole(L"default")->getMaterial(0)->setVector("refractorPosition", D3DXVECTOR4(refractor->getPosition().x, refractor->getPosition().y,refractor->getPosition().z, 1));
	refractor->getMesh()->getRole(L"createPhotonMap")->getMaterial(0)->setVector("refractorPosition", D3DXVECTOR4(refractor->getPosition().x, refractor->getPosition().y,refractor->getPosition().z, 1));

	D3DXMATRIX  sphereModelMatrix, sphereModelMatrixInverse;
	((RigidBody*)refractor)->getModelMatrix(sphereModelMatrix);
	((RigidBody*)refractor)->getModelMatrixInverse(sphereModelMatrixInverse);
	D3DXMatrixInverse(&sphereModelMatrixInverse, NULL, &sphereModelMatrix);
	
	core->getEffect()->SetMatrix("sphereModelMatrix", &sphereModelMatrix);
	core->getEffect()->SetMatrix("sphereModelMatrixInverse", &sphereModelMatrixInverse);

	
	//we recalculate caustics and the height map of the water in every frame
	time += dt;
	fullScreenQuad->getMesh()->getRole(L"createWaterHeightMap")->getMaterial(0)->setVector("Time", D3DXVECTOR4(time, 0, 0, 0));
	renderHeightMap();
	shadedMesh->getRole(L"default")->getMaterial(0)->setTexture("waterHeightMap", heightMap);
	refractor->getMesh()->getRole(L"default")->getMaterial(0)->setTexture("waterHeightMap", heightMap);
	refractor->getMesh()->getRole(L"createPhotonMap")->getMaterial(0)->setTexture("waterHeightMap", heightMap);
	
	renderCaustics();
}

void EnvironmentMappedEntity::handleMessage(HWND hWnd,  UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if(uMsg == WM_KEYDOWN)
	{
		D3DXVECTOR3 pos = refractor->getPosition();
		float offset = 2.0f;
		
		switch(wParam)
		{
			case 'Z':
				refractor->setPosition(D3DXVECTOR3(pos.x, pos.y + offset, pos.z));
				updateRefractorPosition = true;
				break;
			case 'H':
				refractor->setPosition(D3DXVECTOR3(pos.x, pos.y - offset, pos.z));
				updateRefractorPosition = true;
				break;
			case 'F':
				refractor->setPosition(D3DXVECTOR3(pos.x + offset, pos.y , pos.z));
				updateRefractorPosition = true;
				break;
			case 'G':
				refractor->setPosition(D3DXVECTOR3(pos.x - offset, pos.y , pos.z));
				updateRefractorPosition = true;
				break;
			case 'J':
				refractor->setPosition(D3DXVECTOR3(pos.x, pos.y , pos.z + offset));
				updateRefractorPosition = true;
				break;
			case 'K':
				refractor->setPosition(D3DXVECTOR3(pos.x, pos.y , pos.z - offset));
				updateRefractorPosition = true;
				break;
			case 'X':
				if(fresnelFactorWater <= 0.95)
				{
					fresnelFactorWater += 0.05;
					updateFresnelWater = true;
				}
				break;
			case 'Y':
				if(fresnelFactorWater >= 0.05)
				{	
					fresnelFactorWater -= 0.05;
					updateFresnelWater = true;
				}
				break;
			case 'V':
				if(fresnelFactorRefractor <= 0.95)
				{
					fresnelFactorRefractor += 0.05;
					updateFresnelRefractor = true;
				}
				break;
			case 'C':
				if(fresnelFactorRefractor >= 0.05)
				{	
					fresnelFactorRefractor -= 0.05;
					updateFresnelRefractor = true;
				}
				break;
			case 'N':
				indexOfRefractionRefractor += 0.003;
				updateRefractorIOR = true;
				break;
			case 'B':
				if(indexOfRefractionRefractor >= 0.003)
				{	
					indexOfRefractionRefractor -= 0.003;
					updateRefractorIOR = true;
				}
				break;
			case 'L':
				if(fresnelFactorGlass >= 0.05)
				{	
					fresnelFactorGlass -= 0.05;
					updateFresnelGlass = true;
				}
				break;
			case 'O':
				if(fresnelFactorGlass <= 0.95)
				{
					fresnelFactorGlass += 0.05;
					updateFresnelGlass = true;
				}
				break;
			case 'M':
				disableForces = !disableForces;
				break;
		}
	}
	else if(uMsg == WM_KEYUP)
	{
		switch(wParam)
		{
			case 'Z':
				updateRefractorPosition = false;
				break;
			case 'H':
				updateRefractorPosition = false;
				break;
			case 'F':
				updateRefractorPosition = false;
				break;
			case 'G':
				updateRefractorPosition = false;
				break;
			case 'J':
				updateRefractorPosition = false;
				break;
			case 'K':
				updateRefractorPosition = false;
				break;
		}
	}
}
