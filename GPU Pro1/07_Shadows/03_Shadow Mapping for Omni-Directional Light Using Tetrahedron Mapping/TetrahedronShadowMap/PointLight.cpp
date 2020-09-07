//--------------------------------------------------------------------------------------
// File: PointLight.cpp
//
// Encapsulates a point light in the scene by grouping its world matrix with the light
//	parameter.
//
// Copyright (c) Hung-Chien Liao. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "DXUTcamera.h"
#include "PointLight.h"
#include "Obj.h"

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
const float CPointLight::TSMFaceFOVX = 143.985709f + 1.0f;
const float CPointLight::TSMFaceFOVY = 125.264389f + 1.0f;
const int CPointLight::ShadowMapSize = 1024;
const int CPointLight::CubeShadowMapSize = 512;
extern ID3DXEffect* g_pEffect;
extern CFirstPersonCamera g_VCamera;
extern LPDIRECT3DTEXTURE9 g_pShadowMap1;
extern LPDIRECT3DTEXTURE9 g_pShadowMap2;
extern LPDIRECT3DTEXTURE9 g_pHardwareSM1;
extern LPDIRECT3DTEXTURE9 g_pHardwareSM2;
extern LPDIRECT3DCUBETEXTURE9 g_pCubeShadowMap;
extern LPDIRECT3DCUBETEXTURE9 g_pCubeToTSM;
extern LPDIRECT3DSURFACE9 g_pDSShadow;
extern LPDIRECT3DSURFACE9 g_pDSCubeShadow;
extern bool g_bTSMStencil;
extern bool g_bHardwareShadowSupport;
extern bool g_bHardwareShadow;

D3DXMATRIX g_mLightToLightFaceViewProj[6]; // Use to transform from local light space into each light face clip space
D3DXMATRIX g_mWorldToLightFaceViewProj[6]; // Use to transform from world space into each light face clip space


//--------------------------------------------------------------------------------------
// Public Functions
//--------------------------------------------------------------------------------------
DWORD F2DW( FLOAT f ) { return *((DWORD*)&f); }

// Get the objects inside the light range
void CalObjsInLight(std::list<CObj*>& objsInLight, CObj* allObjs, unsigned int numObjs, const CSphere& lightRange)
{
	for (unsigned int i = 0; i < numObjs; ++i)
	{
		if (lightRange.Collision(allObjs[i].m_WorldBound))
			objsInLight.push_back(&allObjs[i]);
	}	
}

// Setup tetrahedron shadow map texture transform
void SetTSMTexTransform()
{
	// The matrix that transform point into final perspective shadow map sapace
	D3DXMATRIX mTexTransform;
	if (g_bTSMStencil)
	{
		D3DXMATRIX mTexScaleBias(0.5f,	0.0f,	0.0f,	0.0f,
								 0.0f,	-0.25f,	0.0f,	0.0f,
								 0.0f,	0.0f,	1.0f,	0.0f,
								 0.5f + (0.5f / CPointLight::ShadowMapSize)/*OffsetX*/,
								 0.75f + (0.25f / CPointLight::ShadowMapSize)/*OffsetY*/,
								 0.0f/*Depth Bias*/,	1.0f);
		// Setup the transform matrix from object space into shadowmap texture space for each Tetrahedron face
		// ...Setup the first face
		D3DXMatrixMultiply(&mTexTransform, &g_mLightToLightFaceViewProj[0], &mTexScaleBias);
		g_pEffect->SetMatrix("g_mTexTransform[0]", &mTexTransform);
		// ...Setup the second face
		mTexScaleBias.m[3][1] = 0.25f + (0.25f / CPointLight::ShadowMapSize);
		D3DXMatrixMultiply(&mTexTransform, &g_mLightToLightFaceViewProj[1], &mTexScaleBias);
		g_pEffect->SetMatrix("g_mTexTransform[1]", &mTexTransform);
		// ...Setup the third face
		mTexScaleBias.m[0][0] = 0.25f;
		mTexScaleBias.m[1][1] = -0.5f;
		mTexScaleBias.m[3][0] = 0.75f + (0.25f / CPointLight::ShadowMapSize);
		mTexScaleBias.m[3][1] = 0.5f + (0.5f / CPointLight::ShadowMapSize);
		D3DXMatrixMultiply(&mTexTransform, &g_mLightToLightFaceViewProj[2], &mTexScaleBias);
		g_pEffect->SetMatrix("g_mTexTransform[2]", &mTexTransform);
		// ...Setup the fourth face
		mTexScaleBias.m[3][0] = 0.25f + (0.25f / CPointLight::ShadowMapSize);
		D3DXMatrixMultiply(&mTexTransform, &g_mLightToLightFaceViewProj[3], &mTexScaleBias);
		g_pEffect->SetMatrix("g_mTexTransform[3]", &mTexTransform);
	}
	else
	{
		D3DXMATRIX mTexScaleBias(0.25f,	0.0f,	0.0f,	0.0f,
								 0.0f,	-0.25f,	0.0f,	0.0f,
								 0.0f,	0.0f,	1.0f,	0.0f,
								 0.25f + (0.25f / CPointLight::ShadowMapSize)/*OffsetX*/,
								 0.25f + (0.25f / CPointLight::ShadowMapSize)/*OffsetY*/,
								 0.0f/*Depth Bias*/,	1.0f);
		// Setup the transform matrix from object space into shadowmap texture space for each Tetrahedron face
		// ...Setup the first face
		D3DXMatrixMultiply(&mTexTransform, &g_mLightToLightFaceViewProj[0], &mTexScaleBias);
		g_pEffect->SetMatrix("g_mTexTransform[0]", &mTexTransform);
		// ...Setup the second face
		mTexScaleBias.m[3][0] = 0.75f + (0.25f / CPointLight::ShadowMapSize);
		D3DXMatrixMultiply(&mTexTransform, &g_mLightToLightFaceViewProj[1], &mTexScaleBias);
		g_pEffect->SetMatrix("g_mTexTransform[1]", &mTexTransform);
		// ...Setup the third face
		mTexScaleBias.m[3][0] = 0.25f + (0.25f / CPointLight::ShadowMapSize);
		mTexScaleBias.m[3][1] = 0.75f + (0.25f / CPointLight::ShadowMapSize);
		D3DXMatrixMultiply(&mTexTransform, &g_mLightToLightFaceViewProj[2], &mTexScaleBias);
		g_pEffect->SetMatrix("g_mTexTransform[2]", &mTexTransform);
		// ...Setup the fourth face
		mTexScaleBias.m[3][0] = 0.75f + (0.25f / CPointLight::ShadowMapSize);
		D3DXMatrixMultiply(&mTexTransform, &g_mLightToLightFaceViewProj[3], &mTexScaleBias);
		g_pEffect->SetMatrix("g_mTexTransform[3]", &mTexTransform);
	}
}

// convert to view matrix
void ConvertToViewMatrix(D3DXMATRIX &mOutView, const D3DXMATRIX &mIn)
{
	D3DXMATRIX mTemp(mIn);
	// Right vector for view matrix
	mOutView.m[0][0] = mTemp.m[0][0];
	mOutView.m[1][0] = mTemp.m[0][1];
	mOutView.m[2][0] = mTemp.m[0][2];
	// Up vector for view matrix
	mOutView.m[0][1] = mTemp.m[1][0];
	mOutView.m[1][1] = mTemp.m[1][1];
	mOutView.m[2][1] = mTemp.m[1][2];
	// Lookat vector for view matrix
	mOutView.m[0][2] = mTemp.m[2][0];
	mOutView.m[1][2] = mTemp.m[2][1];
	mOutView.m[2][2] = mTemp.m[2][2];
	// Position for view matrix
	mOutView.m[3][0] = -D3DXVec3Dot((D3DXVECTOR3*)mTemp.m[0], (D3DXVECTOR3*)mTemp.m[3]);
	mOutView.m[3][1] = -D3DXVec3Dot((D3DXVECTOR3*)mTemp.m[1], (D3DXVECTOR3*)mTemp.m[3]);
	mOutView.m[3][2] = -D3DXVec3Dot((D3DXVECTOR3*)mTemp.m[2], (D3DXVECTOR3*)mTemp.m[3]);

	mOutView.m[0][3] = 0.0f;
	mOutView.m[1][3] = 0.0f;
	mOutView.m[2][3] = 0.0f;
	mOutView.m[3][3] = 1.0f;
}

LRESULT CPointLight::HandleMessages( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
	UNREFERENCED_PARAMETER( hWnd );
    UNREFERENCED_PARAMETER( lParam );

    switch( uMsg )
    {
        case WM_KEYDOWN:
        {
			switch( wParam )
            {
                case 'W':
				{
					m_mWorld.m[3][2] += 0.1f;
					m_WorldBound.m_vCenter.z += 0.1f;
					if (m_pObj)
					{
						m_pObj->m_mWorld.m[3][2] += 0.1f;
						m_pObj->m_WorldBound.m_vCenter.z += 0.1f;
					}
					break;
				}
				case 'S':
				{
					m_mWorld.m[3][2] -= 0.1f;
					m_WorldBound.m_vCenter.z -= 0.1f;
					if (m_pObj)
					{
						m_pObj->m_mWorld.m[3][2] -= 0.1f;
						m_pObj->m_WorldBound.m_vCenter.z -= 0.1f;
					}
					break;
				}
				case 'A':
				{
					m_mWorld.m[3][0] -= 0.1f;
					m_WorldBound.m_vCenter.x -= 0.1f;
					if (m_pObj)
					{
						m_pObj->m_mWorld.m[3][0] -= 0.1f;
						m_pObj->m_WorldBound.m_vCenter.x -= 0.1f;
					}
					break;
				}
				case 'D':
				{
					m_mWorld.m[3][0] += 0.1f;
					m_WorldBound.m_vCenter.x += 0.1f;
					if (m_pObj)
					{
						m_pObj->m_mWorld.m[3][0] += 0.1f;
						m_pObj->m_WorldBound.m_vCenter.x += 0.1f;
					}
					break;
				}
				case 'Q':
				{
					m_mWorld.m[3][1] -= 0.1f;
					m_WorldBound.m_vCenter.y -= 0.1f;
					if (m_pObj)
					{
						m_pObj->m_mWorld.m[3][1] -= 0.1f;
						m_pObj->m_WorldBound.m_vCenter.y -= 0.1f;
					}
					break;
				}
				case 'E':
				{
					m_mWorld.m[3][1] += 0.1f;
					m_WorldBound.m_vCenter.y += 0.1f;
					if (m_pObj)
					{
						m_pObj->m_mWorld.m[3][1] += 0.1f;
						m_pObj->m_WorldBound.m_vCenter.y += 0.1f;
					}
					break;
				}
            }
            break;
        }
    }

    return FALSE;
}

void CPointLight::CalTSMViewProjMatrix()
{
	D3DXMATRIX mLightProj;
	D3DXMATRIX mLightFaceView;	

	// Calculate all four face direction light-view-projection matrix for Tetrahedron
	D3DXMatrixPerspectiveFovLH(&mLightProj, D3DXToRadian(TSMFaceFOVY),
		tanf(D3DXToRadian(TSMFaceFOVX) * 0.5f) / tanf(D3DXToRadian(TSMFaceFOVY) * 0.5f), 0.1f, m_Light.Range);

	// ...Calculate the first face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightFaceView, 0.0f, D3DXToRadian(27.3678055f), 0.0f);
	ConvertToViewMatrix(g_mLightToLightFaceViewProj[0], mLightFaceView);
	D3DXMatrixMultiply(&g_mLightToLightFaceViewProj[0], &g_mLightToLightFaceViewProj[0], &mLightProj);
	D3DXMatrixMultiply(&mLightFaceView, &mLightFaceView, &m_mWorld);
	ConvertToViewMatrix(mLightFaceView, mLightFaceView);
	D3DXMatrixMultiply(&g_mWorldToLightFaceViewProj[0], &mLightFaceView, &mLightProj);
	// ...Calculate the second face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightFaceView, D3DXToRadian(180.0f), D3DXToRadian(27.3678055f), 0.0f);
	ConvertToViewMatrix(g_mLightToLightFaceViewProj[1], mLightFaceView);
	D3DXMatrixMultiply(&g_mLightToLightFaceViewProj[1], &g_mLightToLightFaceViewProj[1], &mLightProj);
	D3DXMatrixMultiply(&mLightFaceView, &mLightFaceView, &m_mWorld);
	ConvertToViewMatrix(mLightFaceView, mLightFaceView);
	D3DXMatrixMultiply(&g_mWorldToLightFaceViewProj[1], &mLightFaceView, &mLightProj);
	// ...Calculate the third face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightFaceView, -D3DXToRadian(90.0f), -D3DXToRadian(27.3678055f), 0.0f);
	ConvertToViewMatrix(g_mLightToLightFaceViewProj[2], mLightFaceView);
	D3DXMatrixMultiply(&g_mLightToLightFaceViewProj[2], &g_mLightToLightFaceViewProj[2], &mLightProj);
	D3DXMatrixMultiply(&mLightFaceView, &mLightFaceView, &m_mWorld);
	ConvertToViewMatrix(mLightFaceView, mLightFaceView);
	D3DXMatrixMultiply(&g_mWorldToLightFaceViewProj[2], &mLightFaceView, &mLightProj);
	// ...Calculate the fourth face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightFaceView, D3DXToRadian(90.0f), -D3DXToRadian(27.3678055f), 0.0f);
	ConvertToViewMatrix(g_mLightToLightFaceViewProj[3], mLightFaceView);
	D3DXMatrixMultiply(&g_mLightToLightFaceViewProj[3], &g_mLightToLightFaceViewProj[3], &mLightProj);
	D3DXMatrixMultiply(&mLightFaceView, &mLightFaceView, &m_mWorld);
	ConvertToViewMatrix(mLightFaceView, mLightFaceView);
	D3DXMatrixMultiply(&g_mWorldToLightFaceViewProj[3], &mLightFaceView, &mLightProj);
}

void CPointLight::CalTSMStencilViewProjMatrix()
{
	D3DXMATRIX mLightProjVert, mLightProjHorz;
	D3DXMATRIX mLightView;

	// Calculate all four face direction light-view-projection matrix for Tetrahedron
	D3DXMatrixPerspectiveFovLH(&mLightProjHorz, D3DXToRadian(TSMFaceFOVY),
		tanf(D3DXToRadian(TSMFaceFOVX) * 0.5f) / tanf(D3DXToRadian(TSMFaceFOVY) * 0.5f), 0.1f, m_Light.Range);
	D3DXMatrixPerspectiveFovLH(&mLightProjVert, D3DXToRadian(TSMFaceFOVX),
		tanf(D3DXToRadian(TSMFaceFOVY) * 0.5f) / tanf(D3DXToRadian(TSMFaceFOVX) * 0.5f), 0.1f, m_Light.Range);

	// ...Calculate the first face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightView, 0.0f, D3DXToRadian(27.3678055f), 0.0f);
	ConvertToViewMatrix(g_mLightToLightFaceViewProj[0], mLightView);
	D3DXMatrixMultiply(&g_mLightToLightFaceViewProj[0], &g_mLightToLightFaceViewProj[0], &mLightProjHorz);
	D3DXMatrixMultiply(&mLightView, &mLightView, &m_mWorld);
	ConvertToViewMatrix(mLightView, mLightView);
	D3DXMatrixMultiply(&g_mWorldToLightFaceViewProj[0], &mLightView, &mLightProjHorz);
	// ...Calculate the second face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightView, D3DXToRadian(180.0f), D3DXToRadian(27.3678055f), D3DXToRadian(180.0f));
	ConvertToViewMatrix(g_mLightToLightFaceViewProj[1], mLightView);
	D3DXMatrixMultiply(&g_mLightToLightFaceViewProj[1], &g_mLightToLightFaceViewProj[1], &mLightProjHorz);
	D3DXMatrixMultiply(&mLightView, &mLightView, &m_mWorld);
	ConvertToViewMatrix(mLightView, mLightView);
	D3DXMatrixMultiply(&g_mWorldToLightFaceViewProj[1], &mLightView, &mLightProjHorz);	
	// ...Calculate the third face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightView, -D3DXToRadian(90.0f), -D3DXToRadian(27.3678055f), D3DXToRadian(90.0f));
	//D3DXVec3Scale((D3DXVECTOR3 *)mLightView.m[1], (D3DXVECTOR3 *)mLightView.m[1], -1.0f);
	ConvertToViewMatrix(g_mLightToLightFaceViewProj[2], mLightView);
	D3DXMatrixMultiply(&g_mLightToLightFaceViewProj[2], &g_mLightToLightFaceViewProj[2], &mLightProjVert);
	D3DXMatrixMultiply(&mLightView, &mLightView, &m_mWorld);
	ConvertToViewMatrix(mLightView, mLightView);
	D3DXMatrixMultiply(&g_mWorldToLightFaceViewProj[2], &mLightView, &mLightProjVert);
	// ...Calculate the fourth face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightView, D3DXToRadian(90.0f), -D3DXToRadian(27.3678055f), -D3DXToRadian(90.0f));
	//D3DXVec3Scale((D3DXVECTOR3 *)mLightView.m[1], (D3DXVECTOR3 *)mLightView.m[1], -1.0f);
	ConvertToViewMatrix(g_mLightToLightFaceViewProj[3], mLightView);
	D3DXMatrixMultiply(&g_mLightToLightFaceViewProj[3], &g_mLightToLightFaceViewProj[3], &mLightProjVert);
	D3DXMatrixMultiply(&mLightView, &mLightView, &m_mWorld);
	ConvertToViewMatrix(mLightView, mLightView);
	D3DXMatrixMultiply(&g_mWorldToLightFaceViewProj[3], &mLightView, &mLightProjVert);
}

void CPointLight::CalCubeViewProjMatrix()
{
	D3DXMATRIX mLightProj;
	D3DXMATRIX mLightView;	

	// Calculate all six face direction light-view-projection matrix for cube
	D3DXMatrixPerspectiveFovLH(&mLightProj, D3DXToRadian(90.0f), 1.0f, 1.0f, m_Light.Range);
	// ...Loop through the six faces of the cube map
    for(DWORD i = 0; i < 6; ++i)
    {
        // Standard view that will be overridden below
        D3DXVECTOR3 vEnvEyePt = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
        D3DXVECTOR3 vLookatPt, vUpVec;

        switch(i)
        {
        case D3DCUBEMAP_FACE_POSITIVE_X:
            vLookatPt = D3DXVECTOR3(1.0f, 0.0f, 0.0f);
            vUpVec    = D3DXVECTOR3(0.0f, 1.0f, 0.0f);
            break;
        case D3DCUBEMAP_FACE_NEGATIVE_X:
            vLookatPt = D3DXVECTOR3(-1.0f, 0.0f, 0.0f);
            vUpVec    = D3DXVECTOR3( 0.0f, 1.0f, 0.0f);
            break;
        case D3DCUBEMAP_FACE_POSITIVE_Y:
            vLookatPt = D3DXVECTOR3(0.0f, 1.0f, 0.0f);
            vUpVec    = D3DXVECTOR3(0.0f, 0.0f,-1.0f);
            break;
        case D3DCUBEMAP_FACE_NEGATIVE_Y:
            vLookatPt = D3DXVECTOR3(0.0f,-1.0f, 0.0f);
            vUpVec    = D3DXVECTOR3(0.0f, 0.0f, 1.0f);
            break;
        case D3DCUBEMAP_FACE_POSITIVE_Z:
            vLookatPt = D3DXVECTOR3( 0.0f, 0.0f, 1.0f);
            vUpVec    = D3DXVECTOR3( 0.0f, 1.0f, 0.0f);
            break;
        case D3DCUBEMAP_FACE_NEGATIVE_Z:
            vLookatPt = D3DXVECTOR3(0.0f, 0.0f,-1.0f);
            vUpVec    = D3DXVECTOR3(0.0f, 1.0f, 0.0f);
            break;
        }
		D3DXMatrixLookAtLH(&mLightView, &vEnvEyePt, &vLookatPt, &vUpVec);
		D3DXMatrixTranspose(&mLightView, &mLightView);
		D3DXMatrixMultiply(&mLightView, &mLightView, &m_mWorld);
		ConvertToViewMatrix(mLightView, mLightView);
		D3DXMatrixMultiply(&g_mWorldToLightFaceViewProj[i], &mLightView, &mLightProj);
	}
}

void CPointLight::GenTSM(IDirect3DDevice9* pd3dDevice, std::list<CObj*>& objsInLight)
{
	HRESULT hr;

	if (g_bTSMStencil)
		CalTSMStencilViewProjMatrix();
	else
		CalTSMViewProjMatrix();

	// Save old viewport
    D3DVIEWPORT9 oldViewport;
    pd3dDevice->GetViewport(&oldViewport);
	LPDIRECT3DSURFACE9 pOldRT = NULL;
    V( pd3dDevice->GetRenderTarget( 0, &pOldRT ) );

	// Render the shadow map
	// ...Set render target to shadow map surfaces
    LPDIRECT3DSURFACE9 pShadowSurf;
    if( SUCCEEDED( g_pShadowMap1->GetSurfaceLevel( 0, &pShadowSurf ) ) )
    {
        pd3dDevice->SetRenderTarget( 0, pShadowSurf );
        SAFE_RELEASE( pShadowSurf );
    }
	// ...Set depth stencil
    LPDIRECT3DSURFACE9 pOldDS = NULL;
    if( SUCCEEDED( pd3dDevice->GetDepthStencilSurface( &pOldDS ) ) )
	{
		if (g_bHardwareShadow)
		{
			LPDIRECT3DSURFACE9 pHardwareShadowSurf;
			if( SUCCEEDED( g_pHardwareSM1->GetSurfaceLevel( 0, &pHardwareShadowSurf ) ) )
			{
				pd3dDevice->SetDepthStencilSurface(pHardwareShadowSurf);
				SAFE_RELEASE( pHardwareShadowSurf );
			}
		}
		else
			pd3dDevice->SetDepthStencilSurface( g_pDSShadow );
	}        
	
	pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
	//pd3dDevice->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, F2DW(1.0f));
	//pd3dDevice->SetRenderState(D3DRS_DEPTHBIAS, F2DW(0.00f));
	CDXUTPerfEventGenerator g( DXUT_PERFEVENTCOLOR, L"Tetrahedron Shadow Map" );
	if (g_bHardwareShadow)
	{
		V( g_pEffect->SetTechnique( "RenderHardwareShadow" ) );
	}
	else
	{
		V( g_pEffect->SetTechnique( "RenderShadow" ) );
	}

    // ...Begin the shadow scene
    if( SUCCEEDED( pd3dDevice->BeginScene() ) )
    {
		// ...Render first Tetrahedron face
		// ......Set new, ShadowMap viewport for the first Tetrahedron face
		D3DVIEWPORT9 newViewport;
		newViewport.X = 0;
		newViewport.Height = int(ShadowMapSize * 0.5f);
		newViewport.MinZ = 0.0f;
		newViewport.MaxZ = 1.0f;
		if (g_bTSMStencil)
		{
			newViewport.Y = int(ShadowMapSize * 0.5f);
			newViewport.Width  = int(ShadowMapSize);
			pd3dDevice->SetRenderState(D3DRS_STENCILENABLE, TRUE);
			pd3dDevice->SetRenderState(D3DRS_STENCILFUNC, D3DCMP_EQUAL);
			pd3dDevice->SetRenderState(D3DRS_STENCILPASS, D3DSTENCILOP_KEEP);
		}
		else
		{
			newViewport.Y = 0;
			newViewport.Width  = int(ShadowMapSize * 0.5f);
		}		
		
		pd3dDevice->SetViewport(&newViewport);
		pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00FFFFFF, 1.0f, 0L);
		// ......Render all object into first Tetrahedron face
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
        {
			if (m_pObj != (*iterObj))
				(*iterObj)->GenShadowMap(g_pEffect, g_mWorldToLightFaceViewProj[0]);
        }

		// ...Render second Tetrahedron face
		// ......Set new, ShadowMap viewport for the second Tetrahedron face
		if (!g_bTSMStencil)
			newViewport.X = int(ShadowMapSize * 0.5f);
		newViewport.Y = 0;
		pd3dDevice->SetViewport(&newViewport);
		pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00FFFFFF, 1.0f, 0L);
		// ......Render all object into second Tetrahedron face
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
        {
			if (m_pObj != (*iterObj))
				(*iterObj)->GenShadowMap(g_pEffect, g_mWorldToLightFaceViewProj[1]);
        }

		// ...Render third Tetrahedron face
		// ......Set new, ShadowMap viewport for the third Tetrahedron face
		if (g_bTSMStencil)
		{
			pd3dDevice->SetRenderState(D3DRS_STENCILFUNC, D3DCMP_NOTEQUAL);
			newViewport.X = int(ShadowMapSize * 0.5f);
			newViewport.Width  = int(ShadowMapSize * 0.5f);
			newViewport.Height = int(ShadowMapSize);			
		}
		else
		{
			newViewport.X = 0;
			newViewport.Y = int(ShadowMapSize * 0.5f);
		}
		pd3dDevice->SetViewport(&newViewport);
		if (!g_bTSMStencil)
			pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00FFFFFF, 1.0f, 0L);
		// ......Render all solid object into third Tetrahedron face
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
        {
			if (m_pObj != (*iterObj))
				(*iterObj)->GenShadowMap(g_pEffect, g_mWorldToLightFaceViewProj[2]);
        }

		// ...Render fourth Tetrahedron face
		// ......Set new, ShadowMap viewport for the fourth Tetrahedron face
		if (g_bTSMStencil)
			newViewport.X = 0;
		else
		{
			newViewport.X = int(ShadowMapSize * 0.5f);
			newViewport.Y = int(ShadowMapSize * 0.5f);
		}
		pd3dDevice->SetViewport(&newViewport);
		if (!g_bTSMStencil)
			pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00FFFFFF, 1.0f, 0L);
		// ......Render all solid object into fourth Tetrahedron face
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
        {
			if (m_pObj != (*iterObj))
				(*iterObj)->GenShadowMap(g_pEffect, g_mWorldToLightFaceViewProj[3]);
        }

		V( pd3dDevice->EndScene() );
	}

	// Set back to normal viewport and render target
    if( pOldDS )
    {
        pd3dDevice->SetDepthStencilSurface( pOldDS );
        pOldDS->Release();
    }
	if (g_bTSMStencil)
		pd3dDevice->SetRenderState(D3DRS_STENCILENABLE, FALSE);
	pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
	//pd3dDevice->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, F2DW(0.0f));
	//pd3dDevice->SetRenderState(D3DRS_DEPTHBIAS, F2DW(0.0f));
    pd3dDevice->SetRenderTarget( 0, pOldRT );
    SAFE_RELEASE( pOldRT );
	pd3dDevice->SetViewport(&oldViewport);
}

void CPointLight::GenTSMLookUp(IDirect3DDevice9* pd3dDevice, std::list<CObj*>& objsInLight)
{
	HRESULT hr;

	if (g_bTSMStencil)
		CalTSMStencilViewProjMatrix();
	else
		CalTSMViewProjMatrix();

	// Save old viewport
    D3DVIEWPORT9 oldViewport;
    pd3dDevice->GetViewport(&oldViewport);
	LPDIRECT3DSURFACE9 pOldRT = NULL;
    V( pd3dDevice->GetRenderTarget( 0, &pOldRT ) );

	// Render the shadow map
	// ...Set render target to shadow map surfaces
    LPDIRECT3DSURFACE9 pShadowSurf;
    if( SUCCEEDED( g_pShadowMap1->GetSurfaceLevel( 0, &pShadowSurf ) ) )
    {
        pd3dDevice->SetRenderTarget( 0, pShadowSurf );
        SAFE_RELEASE( pShadowSurf );
    }
	// ...Set depth stencil
    LPDIRECT3DSURFACE9 pOldDS = NULL;
    if( SUCCEEDED( pd3dDevice->GetDepthStencilSurface( &pOldDS ) ) )
	{
		if (g_bHardwareShadow)
		{
			LPDIRECT3DSURFACE9 pHardwareShadowSurf;
			if( SUCCEEDED( g_pHardwareSM1->GetSurfaceLevel( 0, &pHardwareShadowSurf ) ) )
			{
				pd3dDevice->SetDepthStencilSurface(pHardwareShadowSurf);
				SAFE_RELEASE( pHardwareShadowSurf );
			}
		}
		else
			pd3dDevice->SetDepthStencilSurface( g_pDSShadow );
	}        
	
	pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
	V( g_pEffect->SetFloat("g_fLightRangeSquare", m_Light.Range * m_Light.Range) );
	//pd3dDevice->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, F2DW(1.0f));
	//pd3dDevice->SetRenderState(D3DRS_DEPTHBIAS, F2DW(0.00f));
	CDXUTPerfEventGenerator g( DXUT_PERFEVENTCOLOR, L"Tetrahedron Shadow Map" );
	if (g_bHardwareShadow)
	{
		V( g_pEffect->SetTechnique( "RenderDistanceHardwareShadow" ) );
	}
	else
	{
		V( g_pEffect->SetTechnique( "RenderCubeShadow" ) );
	}

	D3DXMATRIX mLightView;
	ConvertToViewMatrix(mLightView, m_mWorld);
    // ...Begin the shadow scene
    if( SUCCEEDED( pd3dDevice->BeginScene() ) )
    {
		// ...Render first Tetrahedron face
		// ......Set new, ShadowMap viewport for the first Tetrahedron face
		D3DVIEWPORT9 newViewport;
		newViewport.X = 0;
		newViewport.Height = int(ShadowMapSize * 0.5f);
		newViewport.MinZ = 0.0f;
		newViewport.MaxZ = 1.0f;
		if (g_bTSMStencil)
		{
			newViewport.Y = int(ShadowMapSize * 0.5f);
			newViewport.Width  = int(ShadowMapSize);
			pd3dDevice->SetRenderState(D3DRS_STENCILENABLE, TRUE);
			pd3dDevice->SetRenderState(D3DRS_STENCILFUNC, D3DCMP_EQUAL);
			pd3dDevice->SetRenderState(D3DRS_STENCILPASS, D3DSTENCILOP_KEEP);
		}
		else
		{
			newViewport.Y = 0;
			newViewport.Width  = int(ShadowMapSize * 0.5f);
		}		
		
		pd3dDevice->SetViewport(&newViewport);
		pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00FFFFFF, 1.0f, 0L);
		// ......Render all object into first Tetrahedron face
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
        {
			if (m_pObj != (*iterObj))
				(*iterObj)->GenCubeShadowMap(g_pEffect, g_mWorldToLightFaceViewProj[0], mLightView);
        }

		// ...Render second Tetrahedron face
		// ......Set new, ShadowMap viewport for the second Tetrahedron face
		if (!g_bTSMStencil)
			newViewport.X = int(ShadowMapSize * 0.5f);
		newViewport.Y = 0;
		pd3dDevice->SetViewport(&newViewport);
		pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00FFFFFF, 1.0f, 0L);
		// ......Render all object into second Tetrahedron face
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
        {
			if (m_pObj != (*iterObj))
				(*iterObj)->GenCubeShadowMap(g_pEffect, g_mWorldToLightFaceViewProj[1], mLightView);
        }

		// ...Render third Tetrahedron face
		// ......Set new, ShadowMap viewport for the third Tetrahedron face
		if (g_bTSMStencil)
		{
			pd3dDevice->SetRenderState(D3DRS_STENCILFUNC, D3DCMP_NOTEQUAL);
			newViewport.X = int(ShadowMapSize * 0.5f);
			newViewport.Width  = int(ShadowMapSize * 0.5f);
			newViewport.Height = int(ShadowMapSize);			
		}
		else
		{
			newViewport.X = 0;
			newViewport.Y = int(ShadowMapSize * 0.5f);
		}
		pd3dDevice->SetViewport(&newViewport);
		if (!g_bTSMStencil)
			pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00FFFFFF, 1.0f, 0L);
		// ......Render all solid object into third Tetrahedron face
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
        {
			if (m_pObj != (*iterObj))
				(*iterObj)->GenCubeShadowMap(g_pEffect, g_mWorldToLightFaceViewProj[2], mLightView);
        }

		// ...Render fourth Tetrahedron face
		// ......Set new, ShadowMap viewport for the fourth Tetrahedron face
		if (g_bTSMStencil)
			newViewport.X = 0;
		else
		{
			newViewport.X = int(ShadowMapSize * 0.5f);
			newViewport.Y = int(ShadowMapSize * 0.5f);
		}
		pd3dDevice->SetViewport(&newViewport);
		if (!g_bTSMStencil)
			pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00FFFFFF, 1.0f, 0L);
		// ......Render all solid object into fourth Tetrahedron face
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
        {
			if (m_pObj != (*iterObj))
				(*iterObj)->GenCubeShadowMap(g_pEffect, g_mWorldToLightFaceViewProj[3], mLightView);
        }

		V( pd3dDevice->EndScene() );
	}

	// Set back to normal viewport and render target
    if( pOldDS )
    {
        pd3dDevice->SetDepthStencilSurface( pOldDS );
        pOldDS->Release();
    }
	if (g_bTSMStencil)
		pd3dDevice->SetRenderState(D3DRS_STENCILENABLE, FALSE);
	pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
	//pd3dDevice->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, F2DW(0.0f));
	//pd3dDevice->SetRenderState(D3DRS_DEPTHBIAS, F2DW(0.0f));
    pd3dDevice->SetRenderTarget( 0, pOldRT );
    SAFE_RELEASE( pOldRT );
	pd3dDevice->SetViewport(&oldViewport);
}

void CPointLight::GenDSM(IDirect3DDevice9* pd3dDevice, std::list<CObj*>& objsInLight)
{
	HRESULT hr;

	// Save old viewport
    D3DVIEWPORT9 oldViewport;
    pd3dDevice->GetViewport(&oldViewport);
	LPDIRECT3DSURFACE9 pOldRT = NULL;
    V( pd3dDevice->GetRenderTarget( 0, &pOldRT ) );

	// Render the shadow map
	// ...Set render target to shadow map surfaces
    LPDIRECT3DSURFACE9 pShadowSurf;
    if( SUCCEEDED( g_pShadowMap1->GetSurfaceLevel( 0, &pShadowSurf ) ) )
    {
        pd3dDevice->SetRenderTarget( 0, pShadowSurf );
        SAFE_RELEASE( pShadowSurf );
    }
	// ...Set depth stencil
    LPDIRECT3DSURFACE9 pOldDS = NULL;
    if( SUCCEEDED( pd3dDevice->GetDepthStencilSurface( &pOldDS ) ) )
	{
		if (g_bHardwareShadow)
		{
			LPDIRECT3DSURFACE9 pHardwareShadowSurf;
			if( SUCCEEDED( g_pHardwareSM1->GetSurfaceLevel( 0, &pHardwareShadowSurf ) ) )
			{
				pd3dDevice->SetDepthStencilSurface(pHardwareShadowSurf);
				SAFE_RELEASE( pHardwareShadowSurf );
			}
		}
		else
			pd3dDevice->SetDepthStencilSurface( g_pDSShadow );
	}        
	
	pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
	//pd3dDevice->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, F2DW(1.0f));
	//pd3dDevice->SetRenderState(D3DRS_DEPTHBIAS, F2DW(0.00f));
	CDXUTPerfEventGenerator g( DXUT_PERFEVENTCOLOR, L"Dual-Paraboloid Shadow Map" );
	if (g_bHardwareShadow)
	{
		V( g_pEffect->SetTechnique( "RenderFrontHardwareShadow" ) );
	}
	else
	{
		V( g_pEffect->SetTechnique( "RenderFrontShadow" ) );
	}

    // ...Begin the shadow scene
    if( SUCCEEDED( pd3dDevice->BeginScene() ) )
    {
		D3DVIEWPORT9 newViewport;
		newViewport.X = 0;
		newViewport.Y = 0;
		newViewport.Width  = int(ShadowMapSize);
		newViewport.Height = int(ShadowMapSize);
		newViewport.MinZ = 0.0f;
		newViewport.MaxZ = 1.0f;		
		pd3dDevice->SetViewport(&newViewport);

		D3DXMATRIX mLightView;
		ConvertToViewMatrix(mLightView, m_mWorld);
		// ...Render front face
		pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00FFFFFF, 1.0f, 0L);
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
        {
			if (m_pObj != (*iterObj))
				(*iterObj)->GenDSM(g_pEffect, mLightView);
        }

		// ...Render back face
		if (g_bHardwareShadow)
		{
			V( g_pEffect->SetTechnique( "RenderBackHardwareShadow" ) );
		}
		else
		{
			V( g_pEffect->SetTechnique( "RenderBackShadow" ) );
		}
		if( SUCCEEDED( g_pShadowMap2->GetSurfaceLevel( 0, &pShadowSurf ) ) )
		{
			pd3dDevice->SetRenderTarget( 0, pShadowSurf );
			SAFE_RELEASE( pShadowSurf );
		}
		if (g_bHardwareShadow)
		{
			LPDIRECT3DSURFACE9 pHardwareShadowSurf;
			if( SUCCEEDED( g_pHardwareSM2->GetSurfaceLevel( 0, &pHardwareShadowSurf ) ) )
			{
				pd3dDevice->SetDepthStencilSurface(pHardwareShadowSurf);
				SAFE_RELEASE( pHardwareShadowSurf );
			}
		}

		pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00FFFFFF, 1.0f, 0L);
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
        {
			if (m_pObj != (*iterObj))
				(*iterObj)->GenDSM(g_pEffect, mLightView);
        }
		V( pd3dDevice->EndScene() );
	}

	// Set back to normal viewport and render target
    if( pOldDS )
    {
        pd3dDevice->SetDepthStencilSurface( pOldDS );
        pOldDS->Release();
    }
	pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
	//pd3dDevice->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, F2DW(0.0f));
	//pd3dDevice->SetRenderState(D3DRS_DEPTHBIAS, F2DW(0.0f));
    pd3dDevice->SetRenderTarget( 0, pOldRT );
    SAFE_RELEASE( pOldRT );
	pd3dDevice->SetViewport(&oldViewport);
}

void CPointLight::GenCubeSM(IDirect3DDevice9* pd3dDevice, std::list<CObj*>& objsInLight)
{
	HRESULT hr;

	CalCubeViewProjMatrix();

	// Save old viewport
    D3DVIEWPORT9 oldViewport;
    pd3dDevice->GetViewport(&oldViewport);
	LPDIRECT3DSURFACE9 pOldRT = NULL;
    V( pd3dDevice->GetRenderTarget( 0, &pOldRT ) );

	// Render the shadow map	
	// ...Set depth stencil
    LPDIRECT3DSURFACE9 pOldDS = NULL;
    if( SUCCEEDED( pd3dDevice->GetDepthStencilSurface( &pOldDS ) ) )
	{
		pd3dDevice->SetDepthStencilSurface( g_pDSCubeShadow );
	}        
	
	pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
	//pd3dDevice->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, F2DW(1.0f));
	//pd3dDevice->SetRenderState(D3DRS_DEPTHBIAS, F2DW(0.00f));
	CDXUTPerfEventGenerator g( DXUT_PERFEVENTCOLOR, L"Cube Shadow Map" );
	V( g_pEffect->SetTechnique( "RenderCubeShadow" ) );
	V( g_pEffect->SetFloat("g_fLightRangeSquare", m_Light.Range * m_Light.Range) );

	D3DXMATRIX mLightView;
	ConvertToViewMatrix(mLightView, m_mWorld);
    // ...Begin the shadow scene
    if( SUCCEEDED( pd3dDevice->BeginScene() ) )
    {
		for (unsigned int i = 0; i < 6; ++i)
		{
			// ...Set render target to shadow map surfaces
			LPDIRECT3DSURFACE9 pShadowSurf;
			if( SUCCEEDED( g_pCubeShadowMap->GetCubeMapSurface((D3DCUBEMAP_FACES)i, 0, &pShadowSurf ) ) )
			{
				pd3dDevice->SetRenderTarget( 0, pShadowSurf );
				SAFE_RELEASE( pShadowSurf );
			}

			pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00FFFFFF, 1.0f, 0L);
			// ......Render all object into first Tetrahedron face
			for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
				iterObj != objsInLight.end(); ++iterObj )
			{
				if (m_pObj != (*iterObj))
					(*iterObj)->GenCubeShadowMap(g_pEffect, g_mWorldToLightFaceViewProj[i], mLightView);
			}
		}
		V( pd3dDevice->EndScene() );
	}

	// Set back to normal viewport and render target
    if( pOldDS )
    {
        pd3dDevice->SetDepthStencilSurface( pOldDS );
        pOldDS->Release();
    }
	pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
	//pd3dDevice->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, F2DW(0.0f));
	//pd3dDevice->SetRenderState(D3DRS_DEPTHBIAS, F2DW(0.0f));
    pd3dDevice->SetRenderTarget( 0, pOldRT );
    SAFE_RELEASE( pOldRT );
	pd3dDevice->SetViewport(&oldViewport);
}

void CPointLight::RenderSceneTSM(IDirect3DDevice9* pd3dDevice, CObj* allObjs, unsigned int numObjs)
{
	HRESULT hr;

	std::list<CObj*> objsInLight;
	CalObjsInLight(objsInLight, allObjs, numObjs, m_WorldBound);
    //
    // Render the shadow map
    //
	GenTSM(pd3dDevice, objsInLight);

	//
    // Now that we have the shadow map, render the scene.
    //
    // Initialize required parameter
	if (g_bHardwareShadow)
	{
		V( g_pEffect->SetTexture( "g_txShadowFront", g_pHardwareSM1 ) );
		V( g_pEffect->SetTechnique( "PointTSMHardware" ) );
	}
	else
	{
		V( g_pEffect->SetTexture( "g_txShadowFront", g_pShadowMap1 ) );
		V( g_pEffect->SetTechnique( "PointTSM" ) );
	}
	V( g_pEffect->SetVector( "g_vLightDiffuse", (D3DXVECTOR4*)&m_Light.Diffuse ) );
	V( g_pEffect->SetVector( "g_vLightAttenuation", &D3DXVECTOR4(m_Light.Attenuation0,
		m_Light.Attenuation1, m_Light.Attenuation2, m_Light.Range) ) );

	SetTSMTexTransform();
	D3DXMATRIX cameraViewProj;
	D3DXMatrixMultiply(&cameraViewProj, g_VCamera.GetViewMatrix(), g_VCamera.GetProjMatrix());

	D3DXMATRIX mLightView;
	ConvertToViewMatrix(mLightView, m_mWorld);
	CDXUTPerfEventGenerator g( DXUT_PERFEVENTCOLOR, L"Scene" );
	if( SUCCEEDED( pd3dDevice->BeginScene() ) )
    {
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
		{
			(*iterObj)->PointLightRender(g_pEffect, cameraViewProj, mLightView);
		}
		V( pd3dDevice->EndScene() );
	}
}

void CPointLight::RenderSceneTSMLookUp(IDirect3DDevice9* pd3dDevice, CObj* allObjs, unsigned int numObjs)
{
	HRESULT hr;

	std::list<CObj*> objsInLight;
	CalObjsInLight(objsInLight, allObjs, numObjs, m_WorldBound);
    //
    // Render the shadow map
    //
	GenTSMLookUp(pd3dDevice, objsInLight);

	//
    // Now that we have the shadow map, render the scene.
    //
    // Initialize required parameter
	if (g_bHardwareShadow)
	{
		V( g_pEffect->SetTexture( "g_txShadowFront", g_pHardwareSM1 ) );
		V( g_pEffect->SetTechnique( "PointTSMLookHardware" ) );
	}
	else
	{
		V( g_pEffect->SetTexture( "g_txShadowFront", g_pShadowMap1 ) );
		V( g_pEffect->SetTechnique( "PointTSMLook" ) );
	}
	V( g_pEffect->SetTexture( "g_txCubeToTSM", g_pCubeToTSM ) );
	V( g_pEffect->SetVector( "g_vLightDiffuse", (D3DXVECTOR4*)&m_Light.Diffuse ) );
	V( g_pEffect->SetVector( "g_vLightAttenuation", &D3DXVECTOR4(m_Light.Attenuation0,
		m_Light.Attenuation1, m_Light.Attenuation2, m_Light.Range) ) );

	SetTSMTexTransform();
	D3DXMATRIX cameraViewProj;
	D3DXMatrixMultiply(&cameraViewProj, g_VCamera.GetViewMatrix(), g_VCamera.GetProjMatrix());

	D3DXMATRIX mLightView;
	ConvertToViewMatrix(mLightView, m_mWorld);
	CDXUTPerfEventGenerator g( DXUT_PERFEVENTCOLOR, L"Scene" );
	if( SUCCEEDED( pd3dDevice->BeginScene() ) )
    {
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
		{
			(*iterObj)->PointLightRender(g_pEffect, cameraViewProj, mLightView);
		}
		V( pd3dDevice->EndScene() );
	}
}

void CPointLight::RenderSceneDSM(IDirect3DDevice9* pd3dDevice, CObj* allObjs, unsigned int numObjs)
{
	HRESULT hr;

	std::list<CObj*> objsInLight;
	CalObjsInLight(objsInLight, allObjs, numObjs, m_WorldBound);
    //
    // Render the shadow map
    //
	GenDSM(pd3dDevice, objsInLight);

	//
    // Now that we have the shadow map, render the scene.
    //
    // Initialize required parameter
	if (g_bHardwareShadow)
	{
		V( g_pEffect->SetTexture( "g_txShadowFront", g_pHardwareSM1 ) );
		V( g_pEffect->SetTexture( "g_txShadowBack", g_pHardwareSM2 ) );
		V( g_pEffect->SetTechnique( "PointDSMHardware" ) );
	}
	else
	{
		V( g_pEffect->SetTexture( "g_txShadowFront", g_pShadowMap1 ) );
		V( g_pEffect->SetTexture( "g_txShadowBack", g_pShadowMap2 ) );
		V( g_pEffect->SetTechnique( "PointDSM" ) );
	}
	V( g_pEffect->SetVector( "g_vLightDiffuse", (D3DXVECTOR4*)&m_Light.Diffuse ) );
	V( g_pEffect->SetVector( "g_vLightAttenuation", &D3DXVECTOR4(m_Light.Attenuation0,
		m_Light.Attenuation1, m_Light.Attenuation2, m_Light.Range) ) );

	D3DXMATRIX cameraViewProj;
	D3DXMatrixMultiply(&cameraViewProj, g_VCamera.GetViewMatrix(), g_VCamera.GetProjMatrix());

	D3DXMATRIX mLightView;
	ConvertToViewMatrix(mLightView, m_mWorld);
	CDXUTPerfEventGenerator g( DXUT_PERFEVENTCOLOR, L"Scene" );
	if( SUCCEEDED( pd3dDevice->BeginScene() ) )
    {
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
		{
			(*iterObj)->PointLightRender(g_pEffect, cameraViewProj, mLightView);
		}
		V( pd3dDevice->EndScene() );
	}
}

void CPointLight::RenderSceneCubeSM(IDirect3DDevice9* pd3dDevice, CObj* allObjs, unsigned int numObjs)
{
	HRESULT hr;

	std::list<CObj*> objsInLight;
	CalObjsInLight(objsInLight, allObjs, numObjs, m_WorldBound);
    //
    // Render the shadow map
    //
	GenCubeSM(pd3dDevice, objsInLight);

	//
    // Now that we have the shadow map, render the scene.
    //
    // Initialize required parameter
	V( g_pEffect->SetTexture( "g_txCubeShadow", g_pCubeShadowMap ) );
	V( g_pEffect->SetTechnique( "PointCubeSM" ) );
	V( g_pEffect->SetVector( "g_vLightDiffuse", (D3DXVECTOR4*)&m_Light.Diffuse ) );
	V( g_pEffect->SetVector( "g_vLightAttenuation", &D3DXVECTOR4(m_Light.Attenuation0,
		m_Light.Attenuation1, m_Light.Attenuation2, m_Light.Range) ) );

	D3DXMATRIX cameraViewProj;
	D3DXMatrixMultiply(&cameraViewProj, g_VCamera.GetViewMatrix(), g_VCamera.GetProjMatrix());

	D3DXMATRIX mLightView;
	ConvertToViewMatrix(mLightView, m_mWorld);
	CDXUTPerfEventGenerator g( DXUT_PERFEVENTCOLOR, L"Scene" );
	if( SUCCEEDED( pd3dDevice->BeginScene() ) )
    {
		for( std::list<CObj*>::iterator iterObj = objsInLight.begin();
			iterObj != objsInLight.end(); ++iterObj )
		{
			(*iterObj)->PointLightRender(g_pEffect, cameraViewProj, mLightView);
		}
		V( pd3dDevice->EndScene() );
	}
}

void CPointLight::RenderLight(IDirect3DDevice9* pd3dDevice)
{
	HRESULT hr = 0;
	if( m_pObj && SUCCEEDED( pd3dDevice->BeginScene() ) )
    {		
		V( g_pEffect->SetTechnique( "RenderLight" ) );
		// Setup the shader constants
		// ...Setup world-camera view-projection matrix
		D3DXMATRIX cameraViewProj;
		D3DXMatrixMultiply(&cameraViewProj, g_VCamera.GetViewMatrix(), g_VCamera.GetProjMatrix());
		D3DXMATRIX mWorldViewProj;	
		D3DXMatrixMultiply(&mWorldViewProj, &m_pObj->m_mWorld, &cameraViewProj);
		g_pEffect->SetMatrix("g_mWorldViewProj", &mWorldViewProj);

		LPD3DXMESH pMesh = m_pObj->m_Mesh.GetMesh();
		UINT cPass;
		V( g_pEffect->Begin( &cPass, 0 ) );
		for( UINT p = 0; p < cPass; ++p )
		{
			V( g_pEffect->BeginPass( p ) );

			for( DWORD i = 0; i < m_pObj->m_Mesh.m_dwNumMaterials; ++i )
			{
				if( m_pObj->m_Mesh.m_pTextures[i] )
					V( g_pEffect->SetTexture( "g_txDiffuse", m_pObj->m_Mesh.m_pTextures[i] ) )
				else
					V( g_pEffect->SetTexture( "g_txDiffuse", NULL ) )
				V( g_pEffect->CommitChanges() );
				V( pMesh->DrawSubset( i ) );
			}
			V( g_pEffect->EndPass() );
		}
		V( g_pEffect->End() );
		V( pd3dDevice->EndScene() );
	}
}

void CPointLight::CreateCubeToTSMCoord(IDirect3DDevice9* pd3dDevice)
{
	D3DXMATRIX mTSMFaceViewProj0, mTSMFaceViewProj1, mTSMFaceViewProj2, mTSMFaceViewProj3;

	D3DXMATRIX mTexScaleBias(0.5f,	0.0f,	0.0f,			0.0f,
							0.0f,	-0.25f,	0.0f,			0.0f,
							0.0f,	0.0f,	1.0f/*range*/,	0.0f,
							0.5f + (0.5f / ShadowMapSize)/*OffsetX*/,
							0.25f + (0.25f / ShadowMapSize)/*OffsetY*/,
							0.0f/*Bias*/,	1.0f);

	D3DXMATRIX mLightProjVert, mLightProjHorz;
	D3DXMATRIX mLightView;
	// Calculate all four face direction light-view-projection matrix for Tetrahedron
	D3DXMatrixPerspectiveFovLH(&mLightProjHorz, D3DXToRadian(TSMFaceFOVY),
		tanf(D3DXToRadian(TSMFaceFOVX) * 0.5f) / tanf(D3DXToRadian(TSMFaceFOVY) * 0.5f), 0.1f, 100.0f);
	D3DXMatrixPerspectiveFovLH(&mLightProjVert, D3DXToRadian(TSMFaceFOVX),
		tanf(D3DXToRadian(TSMFaceFOVY) * 0.5f) / tanf(D3DXToRadian(TSMFaceFOVX) * 0.5f), 0.1f, 100.0f);

	// ...Calculate the first face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightView, 0.0f, D3DXToRadian(27.3678055f), 0.0f);
	ConvertToViewMatrix(mTSMFaceViewProj0, mLightView);
	D3DXMatrixMultiply(&mTSMFaceViewProj0, &mTSMFaceViewProj0, &mLightProjHorz);
	D3DXMatrixMultiply(&mTSMFaceViewProj0, &mTSMFaceViewProj0, &mTexScaleBias);
	// ...Calculate the second face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightView, D3DXToRadian(180.0f), D3DXToRadian(27.3678055f), D3DXToRadian(180.0f));
	ConvertToViewMatrix(mTSMFaceViewProj1, mLightView);
	D3DXMatrixMultiply(&mTSMFaceViewProj1, &mTSMFaceViewProj1, &mLightProjHorz);
	mTexScaleBias.m[3][1] = 0.75f + (0.25f / ShadowMapSize);
	D3DXMatrixMultiply(&mTSMFaceViewProj1, &mTSMFaceViewProj1, &mTexScaleBias);
	// ...Calculate the third face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightView, -D3DXToRadian(90.0f), -D3DXToRadian(27.3678055f), D3DXToRadian(90.0f));
	ConvertToViewMatrix(mTSMFaceViewProj2, mLightView);
	D3DXMatrixMultiply(&mTSMFaceViewProj2, &mTSMFaceViewProj2, &mLightProjVert);
	mTexScaleBias.m[0][0] = 0.25f;
	mTexScaleBias.m[1][1] = -0.5f;
	mTexScaleBias.m[3][0] = 0.25f + (0.25f / ShadowMapSize);
	mTexScaleBias.m[3][1] = 0.5f + (0.5f / ShadowMapSize);
	D3DXMatrixMultiply(&mTSMFaceViewProj2, &mTSMFaceViewProj2, &mTexScaleBias);
	// ...Calculate the fourth face light-view-projection matrix
	D3DXMatrixRotationYawPitchRoll(&mLightView, D3DXToRadian(90.0f), -D3DXToRadian(27.3678055f), -D3DXToRadian(90.0f));
	ConvertToViewMatrix(mTSMFaceViewProj3, mLightView);
	D3DXMatrixMultiply(&mTSMFaceViewProj3, &mTSMFaceViewProj3, &mLightProjVert);
	mTexScaleBias.m[3][0] = 0.75f + (0.25f / ShadowMapSize);
	D3DXMatrixMultiply(&mTSMFaceViewProj3, &mTSMFaceViewProj3, &mTexScaleBias);

	LPDIRECT3DCUBETEXTURE9 lpCubeToTSM = NULL;
	const int CUBE_SIZE = 128;
	pd3dDevice->CreateCubeTexture(CUBE_SIZE, 1, 0, D3DFMT_A16B16G16R16F, D3DPOOL_SYSTEMMEM,
		&lpCubeToTSM, NULL);
	D3DLOCKED_RECT data;
	D3DXVECTOR4 vVertexPos;
	for (int iFace = 0; iFace < 6; ++iFace)
	{
		lpCubeToTSM->LockRect((D3DCUBEMAP_FACES)iFace, 0, &data, NULL, 0);
		LPBYTE lpBits = (LPBYTE)data.pBits;
		for (float fCoordY = CUBE_SIZE * -0.5f + 0.5f; fCoordY < CUBE_SIZE * 0.5f; ++fCoordY)
		{
			D3DXFLOAT16 *pTexels = (D3DXFLOAT16*)lpBits;
			lpBits += data.Pitch;

			for (float fCoordX = CUBE_SIZE * -0.5f + 0.5f; fCoordX < CUBE_SIZE * 0.5f; ++fCoordX, pTexels += 4)
			{
				switch(iFace)
				{
				case D3DCUBEMAP_FACE_POSITIVE_X:
					vVertexPos = D3DXVECTOR4(CUBE_SIZE * 0.5f - 0.5f, -fCoordY, -fCoordX, 1.0f);
					break;
				case D3DCUBEMAP_FACE_NEGATIVE_X:
					vVertexPos = D3DXVECTOR4(CUBE_SIZE * -0.5f + 0.5f, -fCoordY, fCoordX, 1.0f);
					break;
				case D3DCUBEMAP_FACE_POSITIVE_Y:
					vVertexPos = D3DXVECTOR4(fCoordX, CUBE_SIZE * 0.5f - 0.5f, fCoordY, 1.0f);
					break;
				case D3DCUBEMAP_FACE_NEGATIVE_Y:
					vVertexPos = D3DXVECTOR4(fCoordX, CUBE_SIZE * -0.5f + 0.5f, -fCoordY, 1.0f);
					break;
				case D3DCUBEMAP_FACE_POSITIVE_Z:
					vVertexPos = D3DXVECTOR4(fCoordX, -fCoordY, CUBE_SIZE * 0.5f - 0.5f, 1.0f);
					break;
				case D3DCUBEMAP_FACE_NEGATIVE_Z:
					vVertexPos = D3DXVECTOR4(-fCoordX, -fCoordY, CUBE_SIZE * -0.5f + 0.5f, 1.0f);
					break;
				}
				D3DXVECTOR4 vResult1, vResult2;
				// In group 1, we only need to differentiate face 1 and 2
				if (vVertexPos.z > 0.0f)
				{
					D3DXVec4Transform(&vResult1, &vVertexPos, &mTSMFaceViewProj0);
				}
				else
				{
					D3DXVec4Transform(&vResult1, &vVertexPos, &mTSMFaceViewProj1);
				}
				// In group 2, we only need to differentiate face 3 and 4
				if (vVertexPos.x > 0.0f)
				{
					D3DXVec4Transform(&vResult2, &vVertexPos, &mTSMFaceViewProj3);
				}
				else
				{
					D3DXVec4Transform(&vResult2, &vVertexPos, &mTSMFaceViewProj2);
				}
				vResult1.x /= vResult1.w;
				vResult1.y /= vResult1.w;
				vResult1.y += 0.5f;		// We put +0.5 here, then we do not need to do it at shader
				vResult2.x /= vResult2.w;
				vResult2.x += 0.5f;		// We put +0.5 here, then we do not need to do it at shader
				vResult2.y /= vResult2.w;
				// Save group 1 texture coordinate info in Red and Green channel
				D3DXFloat32To16Array(&pTexels[0], &vResult1.x, 1);
				D3DXFloat32To16Array(&pTexels[1], &vResult1.y, 1);
				// Save group 2 texture coordinate info in Blue and Alpha channel
				D3DXFloat32To16Array(&pTexels[2], &vResult2.x, 1);
				D3DXFloat32To16Array(&pTexels[3], &vResult2.y, 1);
			}
		}
		lpCubeToTSM->UnlockRect((D3DCUBEMAP_FACES)iFace, 0);
	}
	D3DXSaveTextureToFile(L"Textures/CubeToTSMCoord.dds", D3DXIFF_DDS, lpCubeToTSM, NULL);
	SAFE_RELEASE(lpCubeToTSM);
}