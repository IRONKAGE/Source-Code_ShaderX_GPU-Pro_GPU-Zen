#include "dxstdafx.h"
#include "RefractorMesh.h"

RefractorMesh::RefractorMesh(LPD3DXMESH mesh) : ShadedMesh(mesh)
{

}

RefractorMesh::RefractorMesh(LPD3DXMESH mesh, LPD3DXBUFFER materialBuffer, unsigned int nSubMeshes, TextureDirectory& textureDirectory, LPDIRECT3DDEVICE9 device)
: ShadedMesh(mesh, materialBuffer, nSubMeshes, textureDirectory, device)
{
	
}

HRESULT RefractorMesh::createDefaultResources(EngineCore* core)
{
	refractorMap = core->CreateCubeTexture(512, D3DFMT_A16B16G16R16F);
	refractorSurface = core->CreateDepthStencilSurface(512, 512);

	RenderSceneIntoCubeMap(core);

	/*creating geometry image and normal map**************************************************/
	
	geometryImageSize = (int)(pow(2.0f,7.0f)) + 1;

	geometryImage = core->CreateTexture(geometryImageSize, geometryImageSize, D3DFMT_A16B16G16R16F);
	normalMap = core->CreateTexture(geometryImageSize, geometryImageSize, D3DFMT_A16B16G16R16F);
	geometryImageStencilSurface = core->CreateDepthStencilSurface(geometryImageSize ,geometryImageSize);
	
	/*creating min-max maps and link map**************************************************/
	
	minMap = core->CreateTexture(2*geometryImageSize-3, geometryImageSize-1, D3DFMT_A16B16G16R16F);
	maxMap = core->CreateTexture(2*geometryImageSize-3, geometryImageSize-1, D3DFMT_A16B16G16R16F);
	linkMap = core->CreateTexture(2*geometryImageSize-3, geometryImageSize-1, D3DFMT_A16B16G16R16F);
	minMaxStencilSurface = core->CreateDepthStencilSurface(2*geometryImageSize-3, geometryImageSize-1);

	renderGeometryImage(core);

	renderMinMaxMaps(core);

	renderLinkMap(core);

	core->getEffect()->SetTexture("geometryImageTexture", geometryImage);
	core->getEffect()->SetTexture("geometryImageNormalMapTexture", normalMap);
	core->getEffect()->SetTexture("minMapTexture", minMap);
	core->getEffect()->SetTexture("maxMapTexture", maxMap);
	core->getEffect()->SetTexture("linkMapTexture", linkMap);

	D3DXVECTOR2 rootNode = D3DXVECTOR2(1.0f - 0.5f /(float)(2*geometryImageSize-3), 1.0f - 0.5f /(float)(geometryImageSize-1));
	float half = 0.5f/((float)geometryImageSize);
	core->getEffect()->SetVector("rootNode", &D3DXVECTOR4(rootNode.x, rootNode.y, 0, 0));
	core->getEffect()->SetVector("halfTexel", &D3DXVECTOR4(half, half, 0, 0));

	return S_OK;
}

HRESULT RefractorMesh::releaseDefaultResources()
{	
	refractorSurface->Release();
	refractorMap->Release();

	geometryImage->Release();
	geometryImageStencilSurface->Release();
	normalMap->Release();
	minMap->Release();
	maxMap->Release();
	minMaxStencilSurface->Release();
	linkMap->Release();

	return S_OK;
}

bool RefractorMesh::RenderSceneIntoCubeMap(EngineCore* core)
{       
	LPDIRECT3DSURFACE9 oldTarget = NULL;
    core->getDevice()->GetRenderTarget( 0, &oldTarget );
    LPDIRECT3DSURFACE9 oldStencil = NULL;
    if( SUCCEEDED( core->getDevice()->GetDepthStencilSurface( &oldStencil ) ) )
    { 
		core->getDevice()->SetDepthStencilSurface( refractorSurface );
    }

	D3DXVECTOR3 vEnvEyePt = D3DXVECTOR3(0, 0, 0);
		
    for(DWORD i=0; i<6; i++)
    {
        
		D3DXVECTOR3 vUpVec, vLookat;

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


        LPDIRECT3DSURFACE9 pFace;
		refractorMap->GetCubeMapSurface((D3DCUBEMAP_FACES)i, 0, &pFace);
		
			
		core->getDevice()->SetRenderTarget (0 , pFace);
		SAFE_RELEASE(pFace);

		
		core->getDevice()->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0);
			
		core->getDevice()->BeginScene();
		
		D3DXMATRIXA16 mProj;
		D3DXMatrixPerspectiveFovLH( &mProj, D3DX_PI * 0.5f, 1.0f, 0.001f, 10000.0f );
        D3DXMATRIX matView;
		D3DXMatrixLookAtLH(&matView, &vEnvEyePt, &vLookat, &vUpVec);
       
		D3DXMATRIX rootNodeTransform;
		D3DXMatrixIdentity(&rootNodeTransform);

		
		core->setGlobalParameters(matView, mProj, vEnvEyePt);
		RenderContext context = RenderContext(core->getDevice(), core->getEffect(), &mProj,  &matView, &rootNodeTransform, L"storeNormalDistance");

		D3DXMATRIX modelMatrix;
		D3DXMatrixIdentity(&modelMatrix);
		D3DXMATRIX modelMatrixInverse;
		D3DXMatrixInverse(&modelMatrixInverse, NULL, &modelMatrix);
		context.effect->SetMatrix("modelMatrixInverse", &modelMatrixInverse);
		D3DXMATRIX modelViewProjMatrix = modelMatrix * matView * mProj;
		context.effect->SetMatrix("modelViewProjMatrix", &modelViewProjMatrix);

		core->getDevice()->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
		
		render(context);
		
		core->getDevice()->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);

		core->getDevice()->EndScene();

    }


	if( oldStencil )
    {
        core->getDevice()->SetDepthStencilSurface( oldStencil );
        SAFE_RELEASE( oldStencil );
    }
    core->getDevice()->SetRenderTarget( 0, oldTarget );
    SAFE_RELEASE( oldTarget );

	return true;
}

void RefractorMesh::renderGeometryImage(EngineCore* core)
{
	LPDIRECT3DSURFACE9 oldRenderTarget = NULL;
    LPDIRECT3DSURFACE9 oldStencilSurface = NULL;

	LPDIRECT3DSURFACE9 geometryImageSurface;
	geometryImage->GetSurfaceLevel(0, &geometryImageSurface);

	LPDIRECT3DSURFACE9 normalMapSurface;
	normalMap->GetSurfaceLevel(0, &normalMapSurface);

	core->getDevice()->GetRenderTarget(0, &oldRenderTarget);
	core->getDevice()->SetRenderTarget(0, geometryImageSurface);
	core->getDevice()->SetRenderTarget(1, normalMapSurface);
    
	if( SUCCEEDED( core->getDevice()->GetDepthStencilSurface( &oldStencilSurface ) ) )
    { 
		core->getDevice()->SetDepthStencilSurface( geometryImageStencilSurface );
    }

	
	float half = ((float)geometryImageSize) / ((float)geometryImageSize-1);	
	core->getFullScreenQuad()->getMesh()->getRole(L"createSphereGeometryImage")->getMaterial(0)->setVector("halfTexel", D3DXVECTOR4(half,half,0,0));
	core->getFullScreenQuad()->getMesh()->getRole(L"createSphereGeometryImage")->getMaterial(0)->setCubeTexture("refractorMapTexture", refractorMap);

	core->getDevice()->Clear( 0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0 );

	if SUCCEEDED( core->getDevice()->BeginScene() )
	{
		RenderContext context = RenderContext(core->getDevice(), core->getEffect(), NULL, NULL, NULL, L"createSphereGeometryImage");
		core->getFullScreenQuad()->getMesh()->render(context);	
	
		core->getDevice()->EndScene();
	}

	//D3DXSaveSurfaceToFile( L"geometryImage.jpg", D3DXIFF_JPG, geometryImageSurface, NULL, NULL);
	//D3DXSaveSurfaceToFile( L"normalMap.jpg", D3DXIFF_JPG, normalMapSurface, NULL, NULL);

	if( oldStencilSurface )
	{
		core->getDevice()->SetDepthStencilSurface(oldStencilSurface);
		SAFE_RELEASE(oldStencilSurface);
	}
	core->getDevice()->SetRenderTarget(0, oldRenderTarget);
	core->getDevice()->SetRenderTarget(1, NULL);
	SAFE_RELEASE(geometryImageSurface);
	SAFE_RELEASE(normalMapSurface);
	SAFE_RELEASE(oldRenderTarget);
}

void RefractorMesh::renderMinMaxMaps(EngineCore* core)
{
	
	/*creating the mipmap chain resources*************************************************************************/
	
	int mipLevels = (int)( log( (float)(geometryImageSize-1) ) / log ( 2.0f ) + 0.5 ) + 1;
	int size = geometryImageSize - 1;
	
	LPDIRECT3DTEXTURE9* minMapMIP = new LPDIRECT3DTEXTURE9[mipLevels];
	LPDIRECT3DTEXTURE9* maxMapMIP = new LPDIRECT3DTEXTURE9[mipLevels];
	LPDIRECT3DSURFACE9 minMaxMIPStencilSurface = core->CreateDepthStencilSurface(size, size);

	for(int i = 0; i<mipLevels; i++)
	{
		minMapMIP[i] = core->CreateTexture(size, size, D3DFMT_A16B16G16R16F);
		maxMapMIP[i] = core->CreateTexture(size, size, D3DFMT_A16B16G16R16F);
		size /= 2;
	}

	/*rendering to the mipmap chain******************************************************************************/
	
	LPDIRECT3DSURFACE9 oldRenderTarget = NULL;
    LPDIRECT3DSURFACE9 oldStencilSurface = NULL;
	
	if( SUCCEEDED( core->getDevice()->GetDepthStencilSurface( &oldStencilSurface ) ) )
    { 
		core->getDevice()->SetDepthStencilSurface( minMaxMIPStencilSurface );
    }
	core->getDevice()->GetRenderTarget(0, &oldRenderTarget);

	
	float half = 0.5f /(float)(geometryImageSize-1);
	
	
	LPDIRECT3DSURFACE9 minSurface, maxSurface;

	
	for(int i=0; i<mipLevels; i++)
	{
		minMapMIP[i]->GetSurfaceLevel(0, &minSurface);
		maxMapMIP[i]->GetSurfaceLevel(0, &maxSurface);

		core->getDevice()->SetRenderTarget(0, minSurface);
		core->getDevice()->SetRenderTarget(1, maxSurface);

		core->getDevice()->Clear( 0, NULL, D3DCLEAR_ZBUFFER | D3DCLEAR_TARGET, D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0 );

		D3DXVECTOR4* offset = new D3DXVECTOR4[4];	//the four offset values define where we need to query the previous map in the mip chain

		if SUCCEEDED( core->getDevice()->BeginScene() )
		{
			if( i==0 )	//To create the bottomest level of the mip chain we need to query the geometry image. 
			{
				float half0 = 0.5f /(float)(geometryImageSize);

				offset[0] = D3DXVECTOR4(0.0f, 0.0f, 0.0f, 0.0f);
				offset[1] = D3DXVECTOR4(2*half0, 0.0f, 0.0f, 0.0f);
				offset[2] = D3DXVECTOR4(0.0f, 2*half0, 0.0f, 0.0f);
				offset[3] = D3DXVECTOR4(2*half0, 2*half0, 0.0f, 0.0f);

				core->getFullScreenQuad()->getMesh()->getRole(L"createMinMaxMaps")->getMaterial(0)->setTexture("minMapTexture", geometryImage);
				core->getFullScreenQuad()->getMesh()->getRole(L"createMinMaxMaps")->getMaterial(0)->setTexture("maxMapTexture", geometryImage);
				core->getFullScreenQuad()->getMesh()->getRole(L"createMinMaxMaps")->getMaterial(0)->setVector("halfTexel", D3DXVECTOR4(1.0f-3*half0, half0, 0, 0));
			}
			else
			{
				offset[0] = D3DXVECTOR4(-0.5*half, -0.5*half, 0.0f, 0.0f);
				offset[1] = D3DXVECTOR4(-0.5*half, 0.5*half, 0.0f, 0.0f);
				offset[2] = D3DXVECTOR4(0.5*half, 0.5*half, 0.0f, 0.0f);
				offset[3] = D3DXVECTOR4(0.5*half, -0.5*half, 0.0f, 0.0f);

				core->getFullScreenQuad()->getMesh()->getRole(L"createMinMaxMaps")->getMaterial(0)->setTexture("minMapTexture", minMapMIP[i-1]);
				core->getFullScreenQuad()->getMesh()->getRole(L"createMinMaxMaps")->getMaterial(0)->setTexture("maxMapTexture", maxMapMIP[i-1]);
				core->getFullScreenQuad()->getMesh()->getRole(L"createMinMaxMaps")->getMaterial(0)->setVector("halfTexel", D3DXVECTOR4(1.0f, half, 0, 0));
			}
			
			core->getEffect()->SetVectorArray("offset", offset, 4);
			

			RenderContext context = RenderContext(core->getDevice(), core->getEffect(), NULL, NULL, NULL, L"createMinMaxMaps");
			core->getFullScreenQuad()->getMesh()->render(context);

			core->getDevice()->EndScene();
		}

		/*if( i==2 )
		{
			D3DXSaveSurfaceToFile( L"minMap.jpg", D3DXIFF_JPG, minSurface, NULL, NULL);
			D3DXSaveSurfaceToFile( L"maxMap.jpg", D3DXIFF_JPG, maxSurface, NULL, NULL);
		}*/

		SAFE_RELEASE(minSurface);
		SAFE_RELEASE(maxSurface);

		half *= 2.0f;
	}

	
	/*copying the mipmap chain into a single texture******************************************************************************/

	core->getDevice()->SetDepthStencilSurface( minMaxStencilSurface );
	minMap->GetSurfaceLevel(0, &minSurface);
	maxMap->GetSurfaceLevel(0, &maxSurface);
	core->getDevice()->SetRenderTarget(0, minSurface);
	core->getDevice()->SetRenderTarget(1, maxSurface);

	core->getDevice()->Clear( 0, NULL, D3DCLEAR_ZBUFFER | D3DCLEAR_TARGET, D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0 );

	float pow = 1, sum = 0;

	core->getFullScreenQuad()->getMesh()->getRole(L"copyMinMaxMipMap")->getMaterial(0)->setVector("halfTexel", D3DXVECTOR4(0.5f /(float)(2*geometryImageSize-3), 0.5f /(float)(geometryImageSize-1),0,0)); 

	//in each iteration we transform a quad to that part of the render target that we want to fill.
	//The shader copies the selected level of the chain onto the render target.  
	for(int i=1; i<=mipLevels; i++)
	{

		pow *= 0.5f;
		if( i==2 ) sum = 1;
		else if (i>2) sum += pow * 4;

		D3DXMATRIXA16 transformMatrix;
		D3DXMatrixIdentity(&transformMatrix);
		D3DXMatrixTranslation(&transformMatrix, -1+pow+sum, -1+2*pow, 0);
		transformMatrix._11 = pow; transformMatrix._22 = 2*pow;
		

		if SUCCEEDED( core->getDevice()->BeginScene() )
		{
			core->getFullScreenQuad()->getMesh()->getRole(L"copyMinMaxMipMap")->getMaterial(0)->setTexture("minMapTexture", minMapMIP[i-1]);
			core->getFullScreenQuad()->getMesh()->getRole(L"copyMinMaxMipMap")->getMaterial(0)->setTexture("maxMapTexture", maxMapMIP[i-1]);

			core->getEffect()->SetMatrix("transformMatrix", &transformMatrix);

			RenderContext context = RenderContext(core->getDevice(), core->getEffect(), NULL, NULL, NULL, L"copyMinMaxMipMap");
			core->getFullScreenQuad()->getMesh()->render(context);

			core->getDevice()->EndScene();
		}
	
	}

	//D3DXSaveSurfaceToFile( L"minMap.jpg", D3DXIFF_JPG, minSurface, NULL, NULL);
	//D3DXSaveSurfaceToFile( L"maxMap.jpg", D3DXIFF_JPG, maxSurface, NULL, NULL);

	SAFE_RELEASE(minSurface);
	SAFE_RELEASE(maxSurface);

	/************************************************************************************************************************/	

	if( oldStencilSurface )
	{
		core->getDevice()->SetDepthStencilSurface(oldStencilSurface);
		SAFE_RELEASE(oldStencilSurface);
	}
	core->getDevice()->SetRenderTarget(0, oldRenderTarget);
	core->getDevice()->SetRenderTarget(1, NULL);
	SAFE_RELEASE(oldRenderTarget);

	for(int i = 0; i<mipLevels; i++)
	{
		minMapMIP[i]->Release();
		maxMapMIP[i]->Release();
	}
	minMaxMIPStencilSurface->Release();
	
}

void RefractorMesh::renderLinkMap(EngineCore* core)
{
	/*creating the mipmap chain resources*************************************************************************/
	
	int mipLevels = (int)( log( (float)(geometryImageSize-1) ) / log ( 2.0f ) + 0.5 ) + 1;
	
	LPDIRECT3DTEXTURE9* linkMapMIP = new LPDIRECT3DTEXTURE9[mipLevels];
	LPDIRECT3DSURFACE9 linkMIPStencilSurface = core->CreateDepthStencilSurface(geometryImageSize-1, geometryImageSize-1);

	for(int i=0, size=1; i<mipLevels; i++)
	{
		linkMapMIP[i] = core->CreateTexture(size, size, D3DFMT_A16B16G16R16F);
		size *= 2;
	}

	LPDIRECT3DSURFACE9 oldRenderTarget = NULL;
    LPDIRECT3DSURFACE9 oldStencilSurface = NULL;
	

	if( SUCCEEDED( core->getDevice()->GetDepthStencilSurface( &oldStencilSurface ) ) )
    { 
		core->getDevice()->SetDepthStencilSurface( linkMIPStencilSurface );

    }
	core->getDevice()->GetRenderTarget(0, &oldRenderTarget);


	
	
	D3DXVECTOR2 half = D3DXVECTOR2(0.5f /(float)(2*geometryImageSize-3), 0.5f /(float)(geometryImageSize-1));

	float scaleX = 2*half.x; float scaleY = 2*half.y;
	float sum = 0;
	
	D3DXMATRIXA16 transformMatrix, texMatrixHit, texMatrix;

	//texMatrix: transforms coordinates from normalized device space to texture space
	D3DXMatrixIdentity(&texMatrix);
	D3DXMatrixTranslation(&texMatrix, 0.5f + half.x, 0.5f + half.y, 0);
	texMatrix._11 = 0.5f; texMatrix._22 = -0.5f;

	core->getFullScreenQuad()->getMesh()->getRole(L"createLinkMap")->getMaterial(0)->setVector("halfTexel", D3DXVECTOR4(half.x, half.y, 0, 0)); 

	LPDIRECT3DSURFACE9 surface;

	for(int i=0; i<mipLevels; i++) 
	{
		if( i > 0 ) core->getDevice()->SetDepthStencilSurface( linkMIPStencilSurface );
		linkMapMIP[i]->GetSurfaceLevel(0, &surface);
		core->getDevice()->SetRenderTarget(0, surface);
		core->getDevice()->Clear( 0, NULL, D3DCLEAR_ZBUFFER | D3DCLEAR_TARGET, D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0 );
		

		D3DXMatrixIdentity(&transformMatrix);
		D3DXMatrixTranslation(&transformMatrix, 1-scaleX-sum, -1+scaleY, 0);
		transformMatrix._11 = scaleX; transformMatrix._22 = scaleY;
		
		
		scaleX *= 2.0f; scaleY *= 2.0f;
		sum += scaleX;

		D3DXMatrixIdentity(&texMatrixHit);
		D3DXMatrixTranslation(&texMatrixHit, 1-scaleX-sum, -1+scaleY, 0);
		texMatrixHit._11 = scaleX; texMatrixHit._22 = scaleY;
		texMatrixHit = texMatrixHit * texMatrix;
		
		if SUCCEEDED( core->getDevice()->BeginScene() )
		{
			core->getEffect()->SetMatrix("texMatrixNode", &(transformMatrix * texMatrix));
			core->getEffect()->SetMatrix("texMatrixHit", &texMatrixHit);
			
			core->getFullScreenQuad()->getMesh()->getRole(L"createLinkMap")->getMaterial(0)->setVector("linkData", D3DXVECTOR4(geometryImageSize, 0.5f/pow(2.0f,i), (float)i, (float)(mipLevels-1)));
			if( i > 0 ) core->getFullScreenQuad()->getMesh()->getRole(L"createLinkMap")->getMaterial(0)->setTexture("geometryImageTexture", linkMapMIP[i-1]);

			RenderContext context = RenderContext(core->getDevice(), core->getEffect(), NULL, NULL, NULL, L"createLinkMap");
			core->getFullScreenQuad()->getMesh()->render(context);

			core->getDevice()->EndScene();
		}
		
		SAFE_RELEASE(surface);






		core->getDevice()->SetDepthStencilSurface( minMaxStencilSurface );
		linkMap->GetSurfaceLevel(0, &surface);
		core->getDevice()->SetRenderTarget(0, surface);			
		if( i==0 ) core->getDevice()->Clear( 0, NULL, D3DCLEAR_ZBUFFER | D3DCLEAR_TARGET, D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0 );
		
		//we copy the the selected texture onto the link map
		if SUCCEEDED( core->getDevice()->BeginScene() )
		{
			core->getFullScreenQuad()->getMesh()->getRole(L"copyLinkMap")->getMaterial(0)->setTexture("minMapTexture", linkMapMIP[i]);
			
			core->getEffect()->SetMatrix("transformMatrix", &transformMatrix);

			RenderContext context = RenderContext(core->getDevice(), core->getEffect(), NULL, NULL, NULL, L"copyLinkMap");
			core->getFullScreenQuad()->getMesh()->render(context);

			core->getDevice()->EndScene();

			//D3DXSaveSurfaceToFile( L"linkMap.jpg", D3DXIFF_JPG, surface, NULL, NULL);
		}

		SAFE_RELEASE(surface);
	}


	if( oldStencilSurface )
	{
		core->getDevice()->SetDepthStencilSurface(oldStencilSurface);
		SAFE_RELEASE(oldStencilSurface);
	}
	core->getDevice()->SetRenderTarget(0, oldRenderTarget);
	SAFE_RELEASE(oldRenderTarget);

	
	for(int i=0; i<mipLevels; i++) linkMapMIP[i]->Release();
	linkMIPStencilSurface->Release();
}