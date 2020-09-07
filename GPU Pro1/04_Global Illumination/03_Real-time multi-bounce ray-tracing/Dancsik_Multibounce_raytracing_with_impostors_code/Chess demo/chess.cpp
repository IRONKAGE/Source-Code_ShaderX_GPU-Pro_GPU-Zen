#include "dxstdafx.h"
#include "Chess.h"


Chess::Chess(LPDIRECT3DDEVICE9 device):EngineInterface(device)
{	
	
	useClassicalMethod = false;
	fresnelFactor = 0.02f;

	square_size = D3DXVECTOR3(4.0f, 0.0f,4.0f);	
	offset = D3DXVECTOR3(0.1f, 0.0f,0.0f);		
	moving_obj_index = 0;		//all pieces are still

	sceneBoundingBoxMin = D3DXVECTOR3(0.0f,0.0f,0.0f);
	sceneBoundingBoxMax = D3DXVECTOR3(0.0f,0.0f,0.0f);
		
	camera.SetViewParams( &D3DXVECTOR3(2, 21.203f, -27.854f),
						  &D3DXVECTOR3(2, 1, 2));
	camera.SetProjParams(3.14f / 3.0f, 1.0, 0.01, 1000.0);
	camera.SetScalers(0.01f, 0.1f);
	

	/*direct light source setup*/
	D3DXVECTOR3 vecDir = D3DXVECTOR3(-1.0f, -1.0f, 0.0f);
	ZeroMemory( &light, sizeof(light) );
	light.Type = D3DLIGHT_DIRECTIONAL;
	light.Diffuse.r = 1.0f;
	light.Diffuse.g = 1.0f;
	light.Diffuse.b = 1.0f;
	D3DXVec3Normalize( (D3DXVECTOR3*)&light.Direction, &vecDir );

	IntensityOfSnippet[0] = 0.05f; IntensityOfSnippet[1] = 0.06f; IntensityOfSnippet[2] = 0.08f;
	IntensityOfSnippet[3] = 0.11f; IntensityOfSnippet[4] = 0.185f; IntensityOfSnippet[5] = 0.27f;
	IntensityOfSnippet[6] = 0.53f; IntensityOfSnippet[7] = 1.00f; IntensityOfSnippet[8] = 1.65f;
	SizeOfSnippet[0] = 0.05f; SizeOfSnippet[1] = 0.10f; SizeOfSnippet[2] = 0.14f;
	SizeOfSnippet[3] = 0.17f; SizeOfSnippet[4] = 0.2f; SizeOfSnippet[5] =   0.32f;
	SizeOfSnippet[6] = 0.64f; SizeOfSnippet[7] =  1.28f; SizeOfSnippet[8] =  2.56f;


    float fOffsetX = 0.5f + (0.5f / (float)512);
    float fOffsetY = 0.5f + (0.5f / (float)512);
    float range = 1;
    float fBias = -0.001f * range;
    TexScaleBiasMatrix = D3DXMATRIX( 0.5f,     0.0f,     0.0f,  0.0f,
                                     0.0f,    -0.5f,     0.0f,  0.0f,
                                     0.0f,     0.0f,     range, 0.0f,
									 fOffsetX, fOffsetY, fBias, 1.0f );

}




bool Chess::GenerateHeightMap(int mesh_iterator)
{

	int object_id = 0;
	for(int i=1;i<OBJECTS_SIZE;i++)
	{
		if(objects[i]->getMeshIndex()==mesh_iterator)
		{
			object_id = i;
			break;
		}
	}

	D3DXVECTOR3 bmin = objects[object_id]->boundingMin;
	D3DXVECTOR3 bmax = objects[object_id]->boundingMax;

	   
	LPDIRECT3DSURFACE9 pRTOld = NULL;
    device->GetRenderTarget( 0, &pRTOld );
    LPDIRECT3DSURFACE9 pDSOld = NULL;
	if( SUCCEEDED( device->GetDepthStencilSurface( &pDSOld ) ) )
    { 
		device->SetDepthStencilSurface( HeightMapStencilSurface );
	}
	LPDIRECT3DSURFACE9 HeightMapSurface;
	HeightMap[mesh_iterator-1]->GetSurfaceLevel( 0, &HeightMapSurface );
	device->SetRenderTarget( 0, HeightMapSurface );

	
	device->Clear( 0L, NULL, D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER, 0x00000000, 1.0f, 0L );
	
	D3DXMATRIX modelMatrix, modelMatrixInverse;
	D3DXMATRIX viewMatrix, projMatrix;

	D3DXVECTOR3 pivotPoint = (bmin + bmax) / 2;

	D3DXMatrixLookAtLH( &viewMatrix, &D3DXVECTOR3(bmax.x + 0.01f, pivotPoint.y, pivotPoint.z), &pivotPoint, &D3DXVECTOR3(0.0f,1.0f,0.0f) );
	D3DXMatrixOrthoLH(&projMatrix, bmax.z - bmin.z, bmax.y - bmin.y, 0.001, bmax.x - bmin.x);	//orthogonal projection for the height map creation
	
	D3DXMatrixIdentity(&modelMatrix);
	D3DXMatrixInverse(&modelMatrixInverse, NULL, &modelMatrix);

	if( SUCCEEDED( device->BeginScene() ) )
	{
		
		effect->SetMatrix("modelViewProjMatrix", &(modelMatrix*viewMatrix*projMatrix));
		effect->SetMatrix("modelMatrixInverse", &modelMatrixInverse);

		effect->SetTechnique("MapRender");

		UINT a;
		effect->Begin(&a, 0);	
		effect->BeginPass(6);			//height map impostor rendering pass		

		for(int j=0; j<objects[object_id]->GetNumberOfSubMeshes(); j++)
		{					
			mesh[mesh_iterator]->DrawSubset(j);
		}	

		effect->EndPass();
		effect->End();

		device->EndScene();

	}
	
	//D3DXSaveSurfaceToFile( L"HeightMap.png", D3DXIFF_PNG,  HeightMapSurface, NULL,NULL);
	
	if( pDSOld )
    {
       device->SetDepthStencilSurface( pDSOld );
       SAFE_RELEASE( pDSOld );
    }
    device->SetRenderTarget( 0, pRTOld );
    SAFE_RELEASE( pRTOld );
	HeightMapSurface->Release();

	return true;

}




void Chess::CreateRefractorMap(D3DXMATRIX ViewMatrix, D3DXMATRIX ProjMatrix, int mesh_iterator)
{

	device->Clear( 0, NULL, D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(255, 0, 0, 0), 1.0f,0 );
	
	D3DXMATRIX ModelMatrix;
	D3DXMATRIX ModelMatrixInverse;
	D3DXMATRIX ModelViewProjMatrix;
	
	if SUCCEEDED( device->BeginScene() )
	{		

		D3DXMatrixIdentity(&ModelMatrix);
		D3DXMatrixInverse(&ModelMatrixInverse, NULL, &ModelMatrix);
		ModelViewProjMatrix = ModelMatrix * ViewMatrix * ProjMatrix;

		effect->SetMatrix("modelMatrixInverse", &ModelMatrixInverse);
		effect->SetMatrix("modelViewProjMatrix", &ModelViewProjMatrix);

		int submesh_num = 1;
		for(int i=0;i<OBJECTS_SIZE;i++)
		{
			if(objects[i]->getMeshIndex()==mesh_iterator)
			{
				submesh_num = objects[i]->GetNumberOfSubMeshes();
				break;
			}
		}

		effect->SetTechnique("MapRender");
		
		UINT a;
		effect->Begin(&a, 0);
		effect->BeginPass(0);		//object distance impostor rendering pass
		
		for(int j=0; j<submesh_num; j++)
		{			
			mesh[mesh_iterator]->DrawSubset(j);
		}

		effect->EndPass();
		effect->End();

		device->EndScene();

	}
  
}




void Chess::RenderFullScreenQuad()
{

	device->SetStreamSource(0, full_screen_quad, 0, sizeof(D3DXVECTOR4));
							device->SetFVF(D3DFVF_XYZW);
							device->DrawPrimitive( D3DPT_TRIANGLESTRIP, 0, 2 );

}




void Chess::CreateEnvironmentMap(D3DXVECTOR3* eye, D3DXMATRIX viewMatrix,D3DXMATRIX projMatrix, int obj_index, int method)
{
	
	device->Clear( 0, NULL, D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(255, 0, 0, 0), 1.0f,0 );
		
	D3DXMATRIX modelMatrix;
	D3DXMATRIX modelViewProjMatrix;
	D3DXMATRIX modelMatrixInverse;
	D3DXMATRIX viewProjMatrixInverse;
	

	if SUCCEEDED( device->BeginScene() )
	{
		
		if( method == RENDER_ENVMAP_INIT )	//we render the diffuse environment (full-screen quad of the sky, chessboard) into the env. impostor
		{

			D3DXMatrixInverse(&viewProjMatrixInverse, NULL, &(viewMatrix*projMatrix));
			D3DXMatrixIdentity(&modelMatrix);
			D3DXVECTOR3 pos = objects[0]->GetTranslation();
			D3DXMatrixTranslation(&modelMatrix, pos.x, pos.y, pos.z);
			modelViewProjMatrix = modelMatrix * viewMatrix * projMatrix;
			
			effect->SetInt("id_moving", moving_obj_index);

			if( moving_obj_index == obj_index ) effect->SetTexture("FilterTexture_moving",lightMap_moving_obj);
			
			effect->SetMatrix("modelMatrix", &modelMatrix);
			effect->SetMatrix("modelViewProjMatrix", &modelViewProjMatrix);
			effect->SetMatrix("viewProjInverseMatrix", &viewProjMatrixInverse);
			effect->SetTexture("skyCubeTexture", SkyCubeMap);
			effect->SetVector("EyePos",&(D3DXVECTOR4(eye->x,eye->y,eye->z,1)));	
			effect->SetTexture("FilterTexture", lightMap);
			effect->CommitChanges();

			effect->SetTechnique("MapRender");

			UINT a;
			effect->Begin(&a, 0);
	
			effect->BeginPass(1);	//envmap background rendering pass
			
			RenderFullScreenQuad();

			effect->EndPass();
				
			
			effect->BeginPass(2);	//diffuse object rendering pass
			
			effect->SetTexture("diffuseMap", textures[0]);
			effect->CommitChanges();

			objects[0]->getMesh()->DrawSubset(0);
			
			effect->EndPass();

			effect->End();

		}
		else if( method == RENDER_ENVMAP )		//we render the glass pieces into the env. impostor
		{

			effect->SetVector("EyePos", &(D3DXVECTOR4(eye->x,eye->y,eye->z,1)));
		
			for(int i=1; i<OBJECTS_SIZE; i++)
			{

				if( i==obj_index ) continue;		//we skip the piece at the reference point
				//we skip the moving piece if we want to separate the dynamic and static environment
				if( i==moving_obj_index && !useClassicalMethod ) continue;	

				D3DXMatrixIdentity(&modelMatrix);
				D3DXVECTOR3 pos = objects[i]->GetTranslation();
				D3DXVECTOR3 bmin = objects[i]->boundingMin;
				D3DXVECTOR3 bmax = objects[i]->boundingMax;
				D3DXMatrixTranslation(&modelMatrix,pos.x,pos.y,pos.z);
				D3DXMatrixInverse(&modelMatrixInverse, NULL, &modelMatrix);
				modelViewProjMatrix = modelMatrix * viewMatrix * projMatrix;
				
				
				effect->SetVector("selected_obj_boundingMin", &(D3DXVECTOR4(bmin.x,bmin.y,bmin.z,0)));
				effect->SetVector("selected_obj_boundingMax", &(D3DXVECTOR4(bmax.x,bmax.y,bmax.z,0)));
				effect->SetVector("selected_obj_position", &(D3DXVECTOR4(pos.x,pos.y,pos.z,0)));
				effect->SetMatrix("modelMatrix", &modelMatrix);
				effect->SetMatrix("modelMatrixInverse", &modelMatrixInverse);
				effect->SetMatrix("modelViewProjMatrix", &modelViewProjMatrix);
				effect->SetTexture("environmentCubeTexture", environmentmap[i-1]);
				effect->SetTexture("diffuseMap", HeightMap[objects[i]->getMeshIndex()-1]);
				effect->SetTexture("refractorCubeTexture", refractormap[objects[i]->getMeshIndex()-1]);
				effect->SetFloat("id", (FLOAT)i);

				UINT a;
				effect->Begin(&a, 0);
				effect->BeginPass(3);		//envmap reflection and refraction rendering pass

				for(int j=0; j<objects[i]->GetNumberOfSubMeshes(); j++)
				{	
					objects[i]->getMesh()->DrawSubset(j);
				}

				effect->EndPass();
				effect->End();

			}

		}

		device->EndScene();

	}

}




void Chess::RenderGUI()
{

	CDXUTTextHelper txtHelper( Font, NULL, 12 );

	txtHelper.Begin();
    txtHelper.SetInsertionPos( 5, 5 );
    txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
    txtHelper.DrawTextLine( DXUTGetFrameStats( true ) ); 
    txtHelper.DrawTextLine( DXUTGetDeviceStats() );

	txtHelper.SetInsertionPos( 5, 40 );
	txtHelper.DrawTextLine( L"Press [SPACE] to start/stop the movement of the chess figure" );

}




void Chess::render()
{

	device->Clear( 0, NULL, D3DCLEAR_TARGET |D3DCLEAR_ZBUFFER,
			       D3DCOLOR_ARGB(255, 45, 50, 170), 1.0f, 0 );

	D3DXMATRIX modelMatrix;
	D3DXMATRIX modelMatrixInverse;
	D3DXMATRIX modelViewProjMatrix; 
	D3DXMATRIX viewMatrix = *(camera.GetViewMatrix());
	D3DXMATRIX projMatrix = *(camera.GetProjMatrix());

	D3DXMATRIX viewProjMatrixInverse;
	D3DXMatrixInverse(&viewProjMatrixInverse, NULL, &(viewMatrix*projMatrix));
	
	D3DXVECTOR3 bmin;
	D3DXVECTOR3 bmax;
	
	if SUCCEEDED( device->BeginScene() )
	{

		const D3DXVECTOR3* eye = camera.GetEyePt();

		effect->SetTexture("skyCubeTexture",SkyCubeMap);
		effect->SetMatrix("viewProjInverseMatrix",&viewProjMatrixInverse);
		effect->SetVector("EyePos", &(D3DXVECTOR4(eye->x,eye->y,eye->z,0)));
		
		if( moving_obj_index > 0 )
		{
			D3DXVECTOR3 movpos = objects[moving_obj_index]->GetTranslation();
			effect->SetVector("moving_obj_position", &(D3DXVECTOR4(movpos.x,movpos.y,movpos.z,1)));
			effect->SetTexture("refractorCubeTexture_moving_obj", refractormap[objects[moving_obj_index]->getMeshIndex()-1]);
			effect->SetTexture("environmentCubeTexture_moving_obj", environmentmap[moving_obj_index-1]);
			D3DXVECTOR3 bmin = objects[moving_obj_index]->boundingMin;
			D3DXVECTOR3 bmax = objects[moving_obj_index]->boundingMax;
			effect->SetVector("moving_obj_boundingMin", &(D3DXVECTOR4(bmin.x,bmin.y,bmin.z,1)));
			effect->SetVector("moving_obj_boundingMax", &(D3DXVECTOR4(bmax.x,bmax.y,bmax.z,1)));
			effect->SetTexture("FilterTexture_moving", lightMap_moving_obj);
		}
		effect->CommitChanges();
		
		effect->SetTechnique("render");

		UINT a;
		effect->Begin(&a, 0);
		effect->BeginPass(0);	//sky rendering pass

		RenderFullScreenQuad();
	
		effect->EndPass();
		effect->End();
		
		
		for(int i=0; i<OBJECTS_SIZE; i++)
		{	
			
			D3DXMatrixIdentity(&modelMatrix);
			D3DXVECTOR3 pos = objects[i]->GetTranslation();
			D3DXMatrixTranslation(&modelMatrix, pos.x, pos.y, pos.z);
			D3DXMatrixInverse(&modelMatrixInverse, NULL, &modelMatrix);
			modelViewProjMatrix = modelMatrix * viewMatrix * projMatrix;
			
			effect->SetMatrix("modelMatrix", &modelMatrix);
			effect->SetMatrix("modelMatrixInverse", &modelMatrixInverse);
			effect->SetMatrix("modelViewProjMatrix", &modelViewProjMatrix);
			bmin = objects[i]->boundingMin;	
			bmax = objects[i]->boundingMax;	
			effect->SetVector("selected_obj_boundingMin", &(D3DXVECTOR4(bmin.x,bmin.y,bmin.z,1)));	
			effect->SetVector("selected_obj_boundingMax", &(D3DXVECTOR4(bmax.x,bmax.y,bmax.z,1)));
			effect->SetVector("selected_obj_position", &(D3DXVECTOR4(pos.x,pos.y,pos.z,1)));
			effect->SetFloat("id", (FLOAT)i);


			if( i == 1 && moving_obj_index > 0 ) {
				effect->SetTexture("FilterTexture", HeightMap[objects[moving_obj_index]->getMeshIndex()-1]);
			}
			if( i > 0 )
			{
				effect->SetTexture("diffuseMap", HeightMap[objects[i]->getMeshIndex()-1]);
				effect->SetTexture("environmentCubeTexture", environmentmap[i-1]);
				effect->SetTexture("refractorCubeTexture", refractormap[objects[i]->getMeshIndex()-1]);
			}
			

			effect->Begin(&a, 0);
		
			if( i==0 )
				effect->BeginPass(3);		//diffuse pass
			else if( useClassicalMethod || moving_obj_index == 0 || moving_obj_index == i )	//approx. raytr. pass (with static environment)
				effect->BeginPass(1);
			else 
				effect->BeginPass(2);			//approx. raytr. pass (with moving piece)
				
			for(int j=objects[i]->GetNumberOfSubMeshes()-1; j>=0; j--)
			{
				if( i==0 )
				{
					effect->SetTexture("diffuseMap", textures[j]);
					effect->SetTexture("FilterTexture_moving", lightMap_moving_obj);
					effect->SetTexture("FilterTexture", lightMap);
					
					effect->SetFloat("id", -j);

					effect->CommitChanges();
				}

				
				objects[i]->getMesh()->DrawSubset(j);
			}

			effect->EndPass();
			effect->End();

		}

		RenderGUI();
    
		device->EndScene();

		

	}
  
}




void Chess::animate(double dt, double t)
{	
	if( moving_obj_index > 0 )
	{	
		D3DXVECTOR3 pos = objects[moving_obj_index]->GetTranslation();
		
		if( pos.x > 15 || pos.x < -15 ) offset.x = -offset.x;
		
		objects[moving_obj_index]->SetTranslation( pos + offset );
		
		if( useClassicalMethod )	//when using the classical method all env. maps and light maps have to be updated.
		{
			LightMapInitialization(RENDER_ALL_LIGHTMAP);
			CubeMapInitialization(false, RENDER_ALL_ENVMAP);
		}
		else	//when we separate the moving piecefrom the static environment, we have to update the light map and env. map of the moving piece. 
		{
			LightMapInitialization(RENDER_JUST_MOVING_LIGHTMAP);
			CubeMapInitialization(false, RENDER_JUST_MOVING_ENVMAP);
		}
	} 

	camera.FrameMove(dt);
}




void Chess::processMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if( uMsg==WM_KEYDOWN && wParam==VK_SPACE )
	{
		int index = 1;

		if( moving_obj_index == 0 )
		{
			moving_obj_index = index;
			LightMapInitialization(RENDER_ALL_LIGHTMAP);
		}
		else 
		{
			GenerateLightMap(moving_obj_index, false);
			moving_obj_index = 0;
		}

		CubeMapInitialization(false, RENDER_ALL_ENVMAP);
	}
	else camera.HandleMessages(hwnd, uMsg, wParam, lParam);
}




bool Chess::RenderSceneIntoCubeMap( int iterator, int method )
{

	if( method == RENDER_REFRACTION_MAP ) {
		if( iterator <= 0 || iterator >= MESH_SIZE ) return false;
	} else {
		if( iterator <= 0 || iterator >= OBJECTS_SIZE ) return false;
	}

    
	D3DXMATRIXA16 mProj;
    D3DXMatrixPerspectiveFovLH( &mProj, D3DX_PI * 0.5f, 1.0f, 0.001f, 1000.0f );	//90 degree fov

        
	LPDIRECT3DSURFACE9 pRTOld = NULL;
    device->GetRenderTarget( 0, &pRTOld );
    LPDIRECT3DSURFACE9 pDSOld = NULL;
    if( SUCCEEDED( device->GetDepthStencilSurface( &pDSOld ) ) )
    { 
		if( method == RENDER_REFRACTION_MAP )
			device->SetDepthStencilSurface( refractorSurface );
		else
			device->SetDepthStencilSurface( environmentSurface );
    }

	
    // Loop through the six faces of the cube map
    for(DWORD i=0; i<6; i++)
    {

		D3DXVECTOR3 vEnvEyePt;

		//the eye will be at the reference point 
		if( method == RENDER_REFRACTION_MAP ) vEnvEyePt = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
		else vEnvEyePt = objects[iterator]->GetTranslation();

		D3DXVECTOR3 vLookatPt, vUpVec, vLookat;

        switch(i)
        {
            case D3DCUBEMAP_FACE_POSITIVE_X:
				vLookat = D3DXVECTOR3(1.0f, 0.0f, 0.0f);
                vUpVec  = D3DXVECTOR3(0.0f, 1.0f, 0.0f);
                break;
            case D3DCUBEMAP_FACE_NEGATIVE_X:
				vLookat = D3DXVECTOR3(-1.0f, 0.0f, 0.0f);
				vUpVec  = D3DXVECTOR3( 0.0f, 1.0f, 0.0f);
                break;
            case D3DCUBEMAP_FACE_POSITIVE_Y:
				vLookat = D3DXVECTOR3(0.0f, 1.0f, 0.0f);
                vUpVec  = D3DXVECTOR3(0.0f, 0.0f, -1.0f);
                break;
            case D3DCUBEMAP_FACE_NEGATIVE_Y:
				vLookat = D3DXVECTOR3(0.0f,-1.0f, 0.0f);
                vUpVec  = D3DXVECTOR3(0.0f, 0.0f, 1.0f);
                break;
            case D3DCUBEMAP_FACE_POSITIVE_Z:
				vLookat = D3DXVECTOR3( 0.0f, 0.0f, 1.0f);
                vUpVec  = D3DXVECTOR3( 0.0f, 1.0f, 0.0f);
                break;
            case D3DCUBEMAP_FACE_NEGATIVE_Z:
				vLookat = D3DXVECTOR3(0.0f, 0.0f, -1.0f);
                vUpVec  = D3DXVECTOR3(0.0f, 1.0f, 0.0f);
                break;
        }

		if( method == RENDER_REFRACTION_MAP ) 
			vLookatPt = vLookat;
		else 
			vLookatPt = vEnvEyePt + vLookat;

        LPDIRECT3DSURFACE9 pFace;

		if( method == RENDER_REFRACTION_MAP )
			refractormap[iterator-1]->GetCubeMapSurface((D3DCUBEMAP_FACES)i, 0, &pFace);
		else {
			environmentmap[iterator-1]->GetCubeMapSurface((D3DCUBEMAP_FACES)i, 0, &pFace);	
		}
		device->SetRenderTarget (0 , pFace);
        SAFE_RELEASE(pFace);

	
        D3DXMATRIX matView;
		D3DXMatrixLookAtLH(&matView, &vEnvEyePt, &vLookatPt, &vUpVec);
       
		
		if( method == RENDER_REFRACTION_MAP )
			CreateRefractorMap(matView,mProj,iterator); 	
		else {
			CreateEnvironmentMap(&vEnvEyePt,matView,mProj,iterator,method);
		}

    }

	if( pDSOld )
    {
        device->SetDepthStencilSurface( pDSOld );
        SAFE_RELEASE( pDSOld );
    }
    device->SetRenderTarget( 0, pRTOld );
    SAFE_RELEASE( pRTOld );

	return true;

}




void Chess::CubeMapInitialization(bool render_refrmap, int render_envmap_options)
{

	if( render_refrmap )	//object heightmap and distance impostor creation
	{
		for(int i=1;i<MESH_SIZE;i++) RenderSceneIntoCubeMap( i, RENDER_REFRACTION_MAP );
		for(int i=1;i<MESH_SIZE;i++) GenerateHeightMap( i );
	}
	if( render_envmap_options == RENDER_ALL_ENVMAP )	//we render all environment maps
	{
		//first we render the sky and chessboard into the envmaps
		for(int i=1;i<OBJECTS_SIZE;i++) RenderSceneIntoCubeMap( i, RENDER_ENVMAP_INIT );
		for(int j=0; j<2; j++)	//to achieve multiple ray bounces we repeat the rendering of the pieces
		{
			//we render the glass pieces into each env. map
			for(int i=1; i<OBJECTS_SIZE; i++) RenderSceneIntoCubeMap( i, RENDER_ENVMAP );
		}
	}
	else if( render_envmap_options == RENDER_JUST_MOVING_ENVMAP )
	{
		RenderSceneIntoCubeMap( moving_obj_index, RENDER_ENVMAP_INIT );
		//we have to render the static pieces into the env. map of the moving piece just once
		//because the other env. maps already contains the entire static environment
		RenderSceneIntoCubeMap( moving_obj_index, RENDER_ENVMAP );	
	}

}




void Chess::LightMapInitialization(int render_lightmap_options)
{
	
	if( render_lightmap_options == RENDER_JUST_MOVING_LIGHTMAP )
	{	//just the light map of the moving object will be updated
		GenerateLightMap( moving_obj_index, true );
	}
	else if( render_lightmap_options == RENDER_ALL_LIGHTMAP )
	{	//all light map will be updated
		if( moving_obj_index==1 || moving_obj_index==2 ) {
			GenerateLightMap( 1, true );
			GenerateLightMap( 2, true );
		}
		else {
			GenerateLightMap( 1, true );
			GenerateLightMap( 2, false );
		}

		for(int i=3; i<OBJECTS_SIZE; i++)
		{
			if( moving_obj_index == i ) GenerateLightMap( i, true );
			else GenerateLightMap( i, false );
		}
	}

}




void Chess::RenderCausticMap(D3DXMATRIX viewMatrix, D3DXMATRIX projMatrix, D3DXVECTOR3 eye, int obj_index)
{

	device->Clear( 0L, NULL, D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER, 0x00000000, 1.0f, 0L );
	
	D3DXMATRIX modelMatrix;
	D3DXMATRIX modelMatrixInverse;
	D3DXMATRIX modelViewProjMatrix; 

	if( SUCCEEDED( device->BeginScene() ) )
	{

		D3DXMatrixIdentity(&modelMatrix);
		D3DXVECTOR3 pos = objects[obj_index]->GetTranslation();
		D3DXMatrixTranslation(&modelMatrix, pos.x, pos.y, pos.z);
		D3DXMatrixInverse(&modelMatrixInverse, NULL, &modelMatrix);
		modelViewProjMatrix = modelMatrix * viewMatrix * projMatrix;

		int PhotonMapSizeExponent = (int)( log( (float)PHOTON_MAP_SIZE ) / log ( 2.0f ) + 0.5 );	// 0..8
		float SnippetIntensity = IntensityOfSnippet[ PhotonMapSizeExponent ];
		
		// The intensity value of one snippet
		float object_radius = objects[obj_index]->GetRadius();
		float POWER = 0.2 * object_radius * object_radius / 
			( SNIPPET_SIZE * SNIPPET_SIZE * 
			  SnippetIntensity * SnippetIntensity );


		D3DXVECTOR3 bmin = objects[obj_index]->boundingMin;
		D3DXVECTOR3 bmax = objects[obj_index]->boundingMax;
		
		effect->SetVector("EyePos", &(D3DXVECTOR4(eye.x,eye.y,eye.z,1)));
		effect->SetTexture("diffuseMap", HeightMap[objects[obj_index]->getMeshIndex()-1]);	
		effect->SetTexture("refractorCubeTexture", refractormap[objects[obj_index]->getMeshIndex()-1]);
		effect->SetVector("selected_obj_position", &(D3DXVECTOR4(pos.x,pos.y,pos.z,0)));
		effect->SetVector("selected_obj_boundingMin", &(D3DXVECTOR4(bmin.x,bmin.y,bmin.z,0)));
		effect->SetVector("selected_obj_boundingMax", &(D3DXVECTOR4(bmax.x,bmax.y,bmax.z,0)));		
		effect->SetFloat("Power", POWER);
		effect->SetMatrix("modelMatrix", &modelMatrix);
		effect->SetMatrix("modelMatrixInverse", &modelMatrixInverse);
		effect->SetMatrix("modelViewProjMatrix", &modelViewProjMatrix);
		effect->SetVector("LightDir", &(D3DXVECTOR4(light.Direction.x,light.Direction.y,light.Direction.z,1)));
				
		effect->SetTechnique("MapRender");

		UINT a;
		effect->Begin(&a, 0);	
		effect->BeginPass(4);		//photon map rendering pass	

		for(int j=0;j<objects[obj_index]->GetNumberOfSubMeshes();j++)
		{					
			objects[obj_index]->getMesh()->DrawSubset(j);
		}	

		effect->EndPass();
		effect->End();

		device->EndScene();

	}

}




void Chess::RenderShadowMap(D3DXMATRIX viewMatrix, D3DXMATRIX projMatrix, int obj_index)
{

	device->Clear( 0L, NULL, D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER, 0x00000000, 1.0f, 0L );
	
	D3DXMATRIX modelMatrix, modelViewProjMatrix;

	if( SUCCEEDED( device->BeginScene() ) )
	{

		D3DXMatrixIdentity(&modelMatrix);
		D3DXVECTOR3 pos = objects[0]->GetTranslation();
		D3DXMatrixTranslation(&modelMatrix, pos.x, pos.y, pos.z);
		modelViewProjMatrix = modelMatrix * viewMatrix * projMatrix;

		effect->SetMatrix("modelViewProjMatrix", &modelViewProjMatrix);
		
		effect->SetTechnique("MapRender");

		UINT a;
		effect->Begin(&a, 0);	
		effect->BeginPass(7);		//shadow map rendering pass

		objects[0]->getMesh()->DrawSubset(0);

		effect->EndPass();
		effect->End();

		
		D3DXMatrixIdentity(&modelMatrix);
		pos = objects[obj_index]->GetTranslation();
		D3DXMatrixTranslation(&modelMatrix, pos.x, pos.y, pos.z);
		modelViewProjMatrix = modelMatrix * viewMatrix * projMatrix;

		effect->SetMatrix("modelViewProjMatrix", &modelViewProjMatrix);

		effect->Begin(&a, 0);
		effect->BeginPass(7);		//shadow map rendering pass		

		for(int j=0; j<objects[obj_index]->GetNumberOfSubMeshes(); j++)
		{	
			objects[obj_index]->getMesh()->DrawSubset(j);
		}

		effect->EndPass();
		effect->End();

		device->EndScene();

	}

}




void Chess::RenderCausticToLightMap(int object_index)
{

	device->Clear( 0L, NULL, D3DCLEAR_ZBUFFER, 0x00000000, 1.0f, 0L );  

	if( SUCCEEDED( device->BeginScene() ) )
	{		

		effect->SetTechnique("MapRender");

		effect->SetFloat( "CausticsIntensity", CausticsIntensity );		
		effect->SetTexture("causticMapTexture", causticMap);
		effect->SetTexture("PowerOfSnippetMaptexture", SnippetTexture);

		UINT cPass;
		effect->Begin( &cPass, 0 );
		effect->BeginPass( 5 );		//photon splatting pass
	
		device->SetStreamSourceFreq(0, D3DSTREAMSOURCE_INDEXEDDATA | PHOTON_MAP_SIZE*PHOTON_MAP_SIZE);
		device->SetStreamSource(0, causticQuadVertexBuffer, 0, sizeof(D3DXVECTOR4));
		device->SetStreamSourceFreq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1);
		device->SetStreamSource(1, causticQuadInstanceBuffer, 0, 2*sizeof(float));
		device->SetVertexDeclaration(causticQuadVertexDecl);
		device->SetIndices(causticQuadIndexBuffer);

		device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2);

		device->SetStreamSourceFreq(0, 1);
		device->SetStreamSource(0, NULL, 0, 0);
		device->SetStreamSourceFreq(1, 1);
		device->SetStreamSource(1, NULL, 0, 0);
		device->SetIndices(0);

		effect->EndPass();
		effect->End();
		
		device->EndScene();

	}

}




void Chess::RenderShadowToLightMap( D3DXMATRIX mViewLight, D3DXMATRIX mProjLight )
{

	device->Clear( 0L, NULL, D3DCLEAR_ZBUFFER, 0x00000000, 1.0f, 0L );  

	if( SUCCEEDED( device->BeginScene() ) )
	{

		D3DXMATRIX mModel, LightModelViewProjTexBias;

		D3DXMatrixIdentity(&mModel);
		D3DXVECTOR3 pos = objects[0]->GetTranslation();
		D3DXMatrixTranslation(&mModel, pos.x, pos.y, pos.z);

		LightModelViewProjTexBias = mModel * mViewLight * mProjLight * TexScaleBiasMatrix;

		device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
		
		effect->SetTechnique( "MapRender" );

		effect->SetMatrix("LightModelViewProjTexBias", &LightModelViewProjTexBias);
		effect->SetTexture("shadowMapTexture", shadowMap);
		effect->SetVector("LightDir", &(D3DXVECTOR4(light.Direction.x,light.Direction.y,light.Direction.z,1)));

		UINT cPass;
		effect->Begin( &cPass, 0 );
		effect->BeginPass( 8 );
		
		objects[0]->getMesh()->DrawSubset(0);

		effect->EndPass();
		effect->End();
		
		device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
		
		device->EndScene();

	}
	
}




bool Chess::GenerateLightMap(int object_id, boolean delete_lightmap)
{
	if(object_id<=0 || object_id>=OBJECTS_SIZE) return false;

	
	D3DXMATRIXA16 mViewLight, mProjLight;
        
	LPDIRECT3DSURFACE9 pRTOld = NULL;
    device->GetRenderTarget( 0, &pRTOld );
    LPDIRECT3DSURFACE9 pDSOld = NULL;
    
	if( SUCCEEDED( device->GetDepthStencilSurface( &pDSOld ) ) )
    { 
		device->SetDepthStencilSurface( causticMapStencilSurface );
    }


	float radius =  1.05 * objects[object_id]->GetRadius();	//radius 
	float diameter = 2* radius;

	D3DXVECTOR3 upVector = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
	D3DXVECTOR3 lookatPoint = objects[object_id]->GetTranslation();
	D3DXVECTOR3 eyePoint = lookatPoint - radius * light.Direction;


	D3DXMatrixLookAtLH( &mViewLight, &eyePoint, &lookatPoint, &upVector );
	D3DXMatrixOrthoLH(&mProjLight, diameter, diameter, 0.005, 100.0);


	LPDIRECT3DSURFACE9 causticMapSurface;
	causticMap->GetSurfaceLevel( 0, &causticMapSurface );
	device->SetRenderTarget( 0, causticMapSurface );

	RenderCausticMap( mViewLight, mProjLight, eyePoint, object_id );	//photon map rendering


	//D3DXSaveSurfaceToFile( L"PhotonMap.png", D3DXIFF_PNG,  causticMapSurface, NULL, NULL);
	causticMapSurface->Release();
	
	/**************************************************************************/
	
	LPDIRECT3DSURFACE9 shadowMapSurface;
	shadowMap->GetSurfaceLevel( 0, &shadowMapSurface );
	device->SetDepthStencilSurface( shadowMapStencilSurface );
	device->SetRenderTarget( 0, shadowMapSurface );	
	
	RenderShadowMap( mViewLight, mProjLight, object_id );		//shadow map rendering

	//D3DXSaveSurfaceToFile( L"shadowmap.jpg", D3DXIFF_JPG,  shadowMapSurface, NULL, NULL);
	shadowMapSurface->Release();

	/**************************************************************************/
	
	LPDIRECT3DSURFACE9 lightMapSurface;
	
	if( object_id == moving_obj_index )
		lightMap_moving_obj->GetSurfaceLevel( 0, &lightMapSurface );
	else 
		lightMap->GetSurfaceLevel( 0, &lightMapSurface );

	device->SetDepthStencilSurface( lightMapStencilSurface );
	device->SetRenderTarget( 0, lightMapSurface );	
	
	if( delete_lightmap ) device->Clear( 0L, NULL, D3DCLEAR_TARGET, 0x00000000, 1.0f, 0L );
	
	RenderShadowToLightMap( mViewLight, mProjLight );
	RenderCausticToLightMap( object_id );			//photon splatting

	//D3DXSaveSurfaceToFile( L"lightmap.png", D3DXIFF_PNG,  lightMapSurface, NULL,NULL);
	lightMapSurface->Release();

	/***************************************************************************/

	if( pDSOld )
    {
       device->SetDepthStencilSurface( pDSOld );
       SAFE_RELEASE( pDSOld );
    }
    device->SetRenderTarget( 0, pRTOld );
    SAFE_RELEASE( pRTOld );

	return true;

}


 

HRESULT Chess::createManagedResources()
{

	D3DXCreateFont(  device,            // D3D device
                     15,               // Height
                     0,                     // Width
                     FW_BOLD,               // Weight
                     1,                     // MipLevels, 0 = autogen mipmaps
                     FALSE,                 // Italic
                     DEFAULT_CHARSET,       // CharSet
                     OUT_DEFAULT_PRECIS,    // OutputPrecision
                     DEFAULT_QUALITY,       // Quality
                     DEFAULT_PITCH | FF_DONTCARE, // PitchAndFamily
                     L"Arial",              // pFaceName
                     &Font );              // ppFont


	device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);


	//wee load the chess pieces and the chessboard
	int submeshnumb[MESH_SIZE];

	D3DXVECTOR3 boundingMin[MESH_SIZE];
	D3DXVECTOR3 boundingMax[MESH_SIZE];

	LPD3DXBUFFER materialBuffer;

	D3DXLoadMeshFromX(L"Meshes/board.x", D3DXMESH_MANAGED, device, NULL, &materialBuffer, NULL, (DWORD*)&submeshnumb[0], &mesh[0]);
	D3DXLoadMeshFromX(L"Meshes/pawn.x", D3DXMESH_MANAGED, device, NULL, NULL, NULL, (DWORD*)&submeshnumb[1], &mesh[1]);
	D3DXLoadMeshFromX(L"Meshes/bishop.x", D3DXMESH_MANAGED, device, NULL, NULL, NULL, (DWORD*)&submeshnumb[2], &mesh[2]);
	D3DXLoadMeshFromX(L"Meshes/knight.x", D3DXMESH_MANAGED, device, NULL, NULL, NULL, (DWORD*)&submeshnumb[3], &mesh[3]);
	D3DXLoadMeshFromX(L"Meshes/rook.x", D3DXMESH_MANAGED, device, NULL, NULL, NULL, (DWORD*)&submeshnumb[4], &mesh[4]);
	D3DXLoadMeshFromX(L"Meshes/queen.x", D3DXMESH_MANAGED, device, NULL, NULL, NULL, (DWORD*)&submeshnumb[5], &mesh[5]);
	D3DXLoadMeshFromX(L"Meshes/king.x", D3DXMESH_MANAGED, device, NULL, NULL, NULL, (DWORD*)&submeshnumb[6], &mesh[6]);
	D3DXLoadMeshFromX(L"Meshes/sphere.x", D3DXMESH_MANAGED, device, NULL, NULL, NULL, (DWORD*)&submeshnumb[7], &mesh[7]);

	/*we search for the textures of the chessboard*/	
	D3DXMATERIAL* d3dxMaterials = (D3DXMATERIAL*)materialBuffer->GetBufferPointer();
    materials = new D3DMATERIAL9[submeshnumb[0]];
    
	if( materials == NULL ) return E_OUTOFMEMORY;
    
	textures  = new LPDIRECT3DTEXTURE9[submeshnumb[0]];
    
	if( textures == NULL )
        return E_OUTOFMEMORY;

    for( int i=0; i<submeshnumb[0]; i++ )
    {
        // Copy the material
        materials[i] = d3dxMaterials[i].MatD3D;

        // Set the ambient color for the material (D3DX does not do this)
        materials[i].Ambient = materials[i].Diffuse;

        textures[i] = NULL;
        if( d3dxMaterials[i].pTextureFilename != NULL && 
            lstrlenA(d3dxMaterials[i].pTextureFilename) > 0 )
        {  

			// Create the texture
			if( FAILED( D3DXCreateTextureFromFileA( device, 
													d3dxMaterials[i].pTextureFilename, 
													&textures[i] ) 
			) )     
				MessageBox(NULL, L"Could not find texture map", L"Meshes.exe", MB_OK);
			
        }
    }

	materialBuffer->Release();
	

	//Calculate normals if necessary
	for(int i=0; i<MESH_SIZE; i++)
	{
		if( !( mesh[i]->GetFVF() & D3DFVF_NORMAL ) )
		{
			ID3DXMesh* pTempMesh;

			 mesh[i]->CloneMeshFVF( mesh[i]->GetOptions(),
									mesh[i]->GetFVF() | D3DFVF_NORMAL,
									device, &pTempMesh );

			D3DXComputeNormals( pTempMesh, NULL );

			mesh[i]->Release();
			mesh[i] = pTempMesh;
		}
	}

	//Compute bounding boxes
	for(int i=0; i<MESH_SIZE; i++)
	{
		D3DXVECTOR3* pData; 
	
		mesh[i]->LockVertexBuffer( 0, (LPVOID*) &pData ) ;

		D3DXComputeBoundingBox( pData,
								mesh[i]->GetNumVertices(),
								D3DXGetFVFVertexSize(mesh[i]->GetFVF()),
								&boundingMin[i],
								&boundingMax[i]
								);

		mesh[i]->UnlockVertexBuffer();
	}

    
	/************************************************/
	
	//Fill the object array

	int obj_iterator = 0;
	
	//chessboard
	objects[0] = new MeshObject(mesh[0],0);
	objects[0]->SetNumberOfSubMeshes(submeshnumb[0]);
	objects[0]->SetBoundingBox(boundingMin[0],boundingMax[0]);
	objects[0]->SetTranslation(D3DXVECTOR3(0,-objects[0]->boundingMax.y,0));
	obj_iterator++;
	//pawns
	for(int i=0;i<16;i++)
	{
		objects[obj_iterator] = new MeshObject(mesh[1],1);
		objects[obj_iterator]->SetNumberOfSubMeshes(submeshnumb[1]);
		objects[obj_iterator]->SetBoundingBox(boundingMin[1],boundingMax[1]);
		if(i<8) objects[obj_iterator]->SetTranslation(D3DXVECTOR3(3.5*square_size.x-i*square_size.x,-objects[obj_iterator]->boundingMin.y,2.5*square_size.z));
		else objects[obj_iterator]->SetTranslation(D3DXVECTOR3(3.5*square_size.x-(i-8)*square_size.x,-objects[obj_iterator]->boundingMin.y,-2.5*square_size.z));
		obj_iterator++;
	}
	//bishops, knights, rooks
	for(int j=1;j<=3;j++)
	{
		for(int i=0;i<4;i++)
		{
			objects[obj_iterator] = new MeshObject(mesh[1+j],1+j);
			objects[obj_iterator]->SetNumberOfSubMeshes(submeshnumb[1+j]);
			objects[obj_iterator]->SetBoundingBox(boundingMin[1+j],boundingMax[1+j]);
			double x,z;
			if( i==0 || i==1 ) x = -j*square_size.x-0.5*square_size.x; else x = j*square_size.x+0.5*square_size.x;
			if( i==0 || i==2 ) z = 3.5*square_size.z; else z = -3.5*square_size.z;
			objects[obj_iterator]->SetTranslation(D3DXVECTOR3(x,-objects[obj_iterator]->boundingMin.y,z));
			obj_iterator++;
		}
	}
	//king, queen
	for(int j=0;j<=1;j++)
	{
		for(int i=0;i<2;i++)
		{
			objects[obj_iterator] = new MeshObject(mesh[5+j],5+j);
			objects[obj_iterator]->SetNumberOfSubMeshes(submeshnumb[5+j]);
			objects[obj_iterator]->SetBoundingBox(boundingMin[5+j],boundingMax[5+j]);
			if(j==0)
			{
				double x,z;
				x = -0.5*square_size.x;
				if( i==0 ) z = 3.5*square_size.z; else z = -3.5*square_size.z;
				objects[obj_iterator]->SetTranslation(D3DXVECTOR3(x,-objects[obj_iterator]->boundingMin.y,z));
			}
			else
			{
				double x,z;
				x = 0.5*square_size.x;
				if( i==0 ) z = 3.5*square_size.z; else z = -3.5*square_size.z;
				objects[obj_iterator]->SetTranslation(D3DXVECTOR3(x,-objects[obj_iterator]->boundingMin.y,z));
			}
			obj_iterator++;
		}
	}
	//sphere
	objects[obj_iterator] = new MeshObject(mesh[7],7);
	objects[obj_iterator]->SetNumberOfSubMeshes(submeshnumb[7]);
	objects[obj_iterator]->SetBoundingBox(boundingMin[7],boundingMax[7]);
	objects[obj_iterator]->SetTranslation(D3DXVECTOR3(0,-objects[obj_iterator]->boundingMin.y,0));
	
	objects[obj_iterator]->SetTranslation(objects[obj_iterator]->GetTranslation()+D3DXVECTOR3(2,0,2));
	objects[1]->SetTranslation(objects[1]->GetTranslation()+D3DXVECTOR3(0,0,-12));


	for(int i=1; i<OBJECTS_SIZE; i++)
	{
		float temp = objects[i]->boundingMax.y + objects[i]->GetTranslation().y;
		if( temp > square_size.y) square_size.y = temp;
	}

	
	//Calculate the scene bounding box min and max coordinates
	sceneBoundingBoxMin = D3DXVECTOR3(objects[0]->boundingMin.x, objects[0]->boundingMin.y, objects[0]->boundingMin.z) + objects[0]->GetTranslation();
	sceneBoundingBoxMax = D3DXVECTOR3(objects[0]->boundingMax.x, objects[0]->boundingMax.y, objects[0]->boundingMax.z) + objects[0]->GetTranslation();
	for(int i=1; i<OBJECTS_SIZE; i++)
	{
		D3DXVECTOR3 bmin = objects[i]->boundingMin + objects[i]->GetTranslation();
		D3DXVECTOR3 bmax = objects[i]->boundingMax + objects[i]->GetTranslation();

		if(bmin.x < sceneBoundingBoxMin.x) sceneBoundingBoxMin = D3DXVECTOR3(bmin.x, sceneBoundingBoxMin.y, sceneBoundingBoxMin.z);
		if(bmin.y < sceneBoundingBoxMin.y) sceneBoundingBoxMin = D3DXVECTOR3(sceneBoundingBoxMin.x, bmin.y, sceneBoundingBoxMin.z);
		if(bmin.z < sceneBoundingBoxMin.z) sceneBoundingBoxMin = D3DXVECTOR3(sceneBoundingBoxMin.x, sceneBoundingBoxMin.y, bmin.z);

		if(bmax.x > sceneBoundingBoxMax.x) sceneBoundingBoxMax = D3DXVECTOR3(bmax.x, sceneBoundingBoxMax.y, sceneBoundingBoxMax.z);
		if(bmax.y > sceneBoundingBoxMax.y) sceneBoundingBoxMax = D3DXVECTOR3(sceneBoundingBoxMax.x, bmax.y, sceneBoundingBoxMax.z);
		if(bmax.z > sceneBoundingBoxMax.z) sceneBoundingBoxMax = D3DXVECTOR3(sceneBoundingBoxMax.x, sceneBoundingBoxMax.y, bmax.z);
	}
	
	return S_OK;

}




void Chess::CreateCausticQuadrilaterals()
{

	D3DVERTEXELEMENT9 causticQuadVertexElements[] =
	{	//xy: quad size, yw: texture coordinates of the quad
		{0, 0, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
		//xy: the position of the quad in texture space (unique for every quad)
		{1, 0, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, 	D3DDECLUSAGE_POSITION, 0},
		D3DDECL_END() 
	};

	device->CreateVertexDeclaration(causticQuadVertexElements, &causticQuadVertexDecl);

	device->CreateVertexBuffer( 4 * sizeof(D3DXVECTOR4),
								D3DUSAGE_DYNAMIC|D3DUSAGE_WRITEONLY,
								D3DFVF_XYZW,
								D3DPOOL_DEFAULT,
								&causticQuadVertexBuffer,
								NULL
							   );

	D3DXVECTOR4* vertexData;
	causticQuadVertexBuffer->Lock(0, 0, (void**)&vertexData, D3DLOCK_DISCARD);
	
	int PhotonMapSizeExponent = (int)(log( (float)PHOTON_MAP_SIZE ) / log ( 2.0f ) + 0.5f );	// 0..8
	float SnippetSize = SizeOfSnippet[ PhotonMapSizeExponent ] * SNIPPET_SIZE / PHOTON_MAP_SIZE;


	//float4(offset.x, offset.y, texcoord.u, texcoord.v)
	vertexData[0] = D3DXVECTOR4(-SnippetSize, -SnippetSize, 0.0f, 0.0f);
	vertexData[1] = D3DXVECTOR4(SnippetSize, -SnippetSize, 1.0f, 0.0f);
	vertexData[2] = D3DXVECTOR4(SnippetSize, SnippetSize, 1.0f, 1.0f);
	vertexData[3] = D3DXVECTOR4(-SnippetSize, SnippetSize, 0.0f, 1.0f);	

	causticQuadVertexBuffer->Unlock();

	//Create the index buffer
	device->CreateIndexBuffer(6 * sizeof(short), 0, D3DFMT_INDEX16, D3DPOOL_DEFAULT, &causticQuadIndexBuffer, NULL);

	short* indexData;
	causticQuadIndexBuffer->Lock(0, 0, (void**)&indexData, D3DLOCK_DISCARD);
	indexData[0] = 0; indexData[1] = 3; indexData[2] = 1; indexData[3] = 2; indexData[4] = 1; indexData[5] = 3;
	causticQuadIndexBuffer->Unlock();

	
	//Create the instance buffer
	device->CreateVertexBuffer(	2 * PHOTON_MAP_SIZE * PHOTON_MAP_SIZE * sizeof(float),
								D3DUSAGE_DYNAMIC, 0,
								D3DPOOL_DEFAULT,
								&causticQuadInstanceBuffer,
								NULL
							   );

	
	float* instanceData;
	causticQuadInstanceBuffer->Lock(0, 0, (void**)&instanceData, D3DLOCK_DISCARD);

	int counter = 0;
	for ( int i = 0; i < PHOTON_MAP_SIZE; i++ )
	{
		for ( int j = 0; j < PHOTON_MAP_SIZE; j++ )
		{
			float u = (float) i / PHOTON_MAP_SIZE;
			float v = (float) j / PHOTON_MAP_SIZE;
			
			instanceData[counter] = u; counter += 1;
			instanceData[counter] = v; counter += 1;
		}
	}
	causticQuadInstanceBuffer->Unlock();

}




LPDIRECT3DCUBETEXTURE9 Chess::CreateCubeTexture( int size, D3DFORMAT Format )
{
	HRESULT hr;
	LPDIRECT3DCUBETEXTURE9 CubeTexture;
	V( device->CreateCubeTexture( size, 1, D3DUSAGE_RENDERTARGET,
								  Format, D3DPOOL_DEFAULT, &CubeTexture, NULL ) );
	return CubeTexture;
}




LPDIRECT3DTEXTURE9 Chess::CreateTexture( int size, D3DFORMAT Format )
{
	HRESULT hr;
	LPDIRECT3DTEXTURE9 Texture;
	V( device->CreateTexture( size, size, 1, D3DUSAGE_RENDERTARGET, 
							  Format, D3DPOOL_DEFAULT, &Texture, NULL ) );
	return Texture;
}




LPDIRECT3DSURFACE9 Chess::CreateDepthStencilSurface( int size )
{
	HRESULT hr;
	LPDIRECT3DSURFACE9 Surface;
	V( device->CreateDepthStencilSurface( size, size, 
										  DXUTGetDeviceSettings().pp.AutoDepthStencilFormat,
                                          D3DMULTISAMPLE_NONE, 0, TRUE, &Surface, NULL ) );
	return Surface;
}




HRESULT Chess::createDefaultResources(wchar_t* effectFileName)
{	
	
	/*Load the effect file*************************************************/
	
	LPD3DXBUFFER compilationErrors;
	
	if( FAILED( D3DXCreateEffectFromFile(device, effectFileName,
										 NULL, NULL, 0, NULL, &effect,
										 &compilationErrors) ) )
	{
		if(compilationErrors)
			MessageBoxA( NULL,(LPSTR)compilationErrors->GetBufferPointer(),
					     "Failed to load effect file!", MB_OK);
		exit(-1);
	}

	effect->SetFloat("FresnelFactor", fresnelFactor);
	effect->SetFloat("IndexOfRefraction",indexOfRefraction);
	effect->SetVector("chesstable_data", &(D3DXVECTOR4(objects[0]->boundingMin.x, objects[0]->boundingMin.z, objects[0]->boundingMax.x, objects[0]->boundingMax.z)));
	effect->CommitChanges();

	/*full-screen_quad***********************************************************/

	device->CreateVertexBuffer( 4*sizeof(D3DXVECTOR4), D3DUSAGE_DYNAMIC|D3DUSAGE_WRITEONLY,
							    D3DFVF_XYZW, D3DPOOL_DEFAULT, &full_screen_quad, NULL );
    
	D3DXVECTOR4* vertexData;
	full_screen_quad->Lock(0, 0, (void**)&vertexData, D3DLOCK_DISCARD);


	vertexData[0] = D3DXVECTOR4(1.0f, -1.0f, 1.0f, 1.0f);
    vertexData[1] = D3DXVECTOR4(1.0f, 1.0f, 1.0f, 1.0f);
    vertexData[2] = D3DXVECTOR4(-1.0f, -1.0f, 1.0f, 1.0f); 
	vertexData[3] = D3DXVECTOR4(-1.0, 1.0, 1.0f, 1.0f);
	
	full_screen_quad->Unlock();
	
	/*Load the sky texture***************************************************/

	D3DXCreateCubeTextureFromFileEx( device, L"uffizi_cross.dds", D3DX_DEFAULT, 1, 0, D3DFMT_A16B16G16R16F, 
                                     D3DPOOL_MANAGED, D3DX_FILTER_NONE, D3DX_FILTER_NONE, 0, NULL, NULL, &SkyCubeMap );
	
	/*Create height map impostors********************************************/
	
	for(int i=0; i<MESH_SIZE-1; i++)
	{
		HeightMap[i] = CreateTexture(512, D3DFMT_A16B16G16R16F);
	}

	HeightMapStencilSurface = CreateDepthStencilSurface(512);

	/*Create object distance impostors********************************************/
	
	for(int i=0; i<MESH_SIZE-1; i++)
	{
		refractormap[i] = CreateCubeTexture(512, D3DFMT_A16B16G16R16F);
	}

	refractorSurface = CreateDepthStencilSurface(512);
	
	/*Create environment impostors********************************************/

	for(int i=0; i<OBJECTS_SIZE-1; i++)
	{	
		environmentmap[i] = CreateCubeTexture(256, D3DFMT_A16B16G16R16F);
	}

	environmentSurface = CreateDepthStencilSurface(256);

	/*Load filter texture*********************************************************/

	D3DXCreateTextureFromFile( device, L"PowerOfSnippetTexel.dds", &SnippetTexture);

	CreateCausticQuadrilaterals();

	/*Create photon map and the two light maps********************************************/

	causticMap = CreateTexture(PHOTON_MAP_SIZE, D3DFMT_A32B32G32R32F);
	causticMapStencilSurface = CreateDepthStencilSurface(PHOTON_MAP_SIZE);

	shadowMap = CreateTexture(512, D3DFMT_R32F);
	shadowMapStencilSurface = CreateDepthStencilSurface(512);


	D3DSURFACE_DESC desc;
	textures[0]->GetLevelDesc(0, &desc);

	
	lightMap = CreateTexture(desc.Width, D3DFMT_A16B16G16R16F);
	lightMap_moving_obj = CreateTexture(desc.Width, D3DFMT_A16B16G16R16F);

	lightMapStencilSurface = CreateDepthStencilSurface(desc.Width);

	/*Render the cube maps nad light maps********************************************/
	
	CubeMapInitialization(true, -1);	//render height maps and object distance impostors
	LightMapInitialization(RENDER_ALL_LIGHTMAP);		//render light maps
	CubeMapInitialization(false, RENDER_ALL_ENVMAP);	//render environment impostors

	return S_OK;

}




HRESULT Chess::releaseManagedResources()
{ 

	for(int i=0;i<MESH_SIZE;i++) mesh[i]->Release();

	if( materials != NULL ) 
		delete[] materials;

	if( textures )
	{
		for( int i=0; i < objects[0]->GetNumberOfSubMeshes(); i++ )
		{
			if( textures[i] )
				textures[i]->Release();
		}
		delete[] textures;
	}

	Font->Release();

	return S_OK;

}



HRESULT Chess::releaseDefaultResources()
{

		effect->Release();
		
		SkyCubeMap->Release();
		
		full_screen_quad->Release();
		
		for(int i=0; i<MESH_SIZE-1; i++) refractormap[i]->Release();
		refractorSurface->Release();

		for(int i=0; i<MESH_SIZE-1; i++) HeightMap[i]->Release();
		HeightMapStencilSurface->Release();

		for(int i=0; i<OBJECTS_SIZE-1; i++) environmentmap[i]->Release();
		environmentSurface->Release();

		causticMap->Release();
		causticMapStencilSurface->Release();

		shadowMap->Release();
		shadowMapStencilSurface->Release();

		lightMap->Release();
		lightMap_moving_obj->Release();
		lightMapStencilSurface->Release();
		
		causticQuadVertexDecl->Release();
		causticQuadVertexBuffer->Release();
		causticQuadIndexBuffer->Release();
		causticQuadInstanceBuffer->Release();

		SnippetTexture->Release();
		
		return S_OK;

}

