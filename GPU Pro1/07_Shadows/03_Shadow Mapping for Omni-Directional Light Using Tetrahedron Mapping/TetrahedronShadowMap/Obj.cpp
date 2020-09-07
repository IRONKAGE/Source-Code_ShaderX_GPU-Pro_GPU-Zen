//--------------------------------------------------------------------------------------
// File: Obj.cpp
//
// Encapsulates a mesh object in the scene by grouping its world matrix with the mesh.
//
// Copyright (c) Hung-Chien Liao. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "Obj.h"

//--------------------------------------------------------------------------------------
// Public Functions
//--------------------------------------------------------------------------------------
void CObj::GenShadowMap(ID3DXEffect* pEffect, const D3DXMATRIXA16& mLightViewProj)
{
	HRESULT hr;
	D3DXMATRIXA16 mWorldViewProj = m_mWorld;
    D3DXMatrixMultiply( &mWorldViewProj, &mWorldViewProj, &mLightViewProj );
    V( pEffect->SetMatrix( "g_mWorldViewProj", &mWorldViewProj ) );

    //LPD3DXMESH pMesh = m_Mesh.GetMesh();
    UINT cPass;
    V( pEffect->Begin( &cPass, 0 ) );
    for( UINT p = 0; p < cPass; ++p )
    {
        V( pEffect->BeginPass( p ) );

        for( DWORD i = 0; i < m_Mesh.m_dwNumMaterials; ++i )
        {
            V( pEffect->CommitChanges() );
            V( m_Mesh.GetMesh()->DrawSubset( i ) );
        }
        V( pEffect->EndPass() );
    }
    V( pEffect->End() );
}

void CObj::GenDSM(ID3DXEffect* pEffect, const D3DXMATRIX& mLightView)
{
	HRESULT hr;
	// ...Setup world-light view matrix
	D3DXMATRIX mWorldLightView;
	D3DXMatrixMultiply(&mWorldLightView, &m_mWorld, &mLightView);
	pEffect->SetMatrix("g_mWorldLightView", &mWorldLightView);

    //LPD3DXMESH pMesh = m_Mesh.GetMesh();
    UINT cPass;
    V( pEffect->Begin( &cPass, 0 ) );
    for( UINT p = 0; p < cPass; ++p )
    {
        V( pEffect->BeginPass( p ) );

        for( DWORD i = 0; i < m_Mesh.m_dwNumMaterials; ++i )
        {
            V( pEffect->CommitChanges() );
            V( m_Mesh.GetMesh()->DrawSubset( i ) );
        }
        V( pEffect->EndPass() );
    }
    V( pEffect->End() );
}

void CObj::GenCubeShadowMap(ID3DXEffect* pEffect, const D3DXMATRIXA16& mLightViewProj, const D3DXMATRIXA16& mLightView)
{
	HRESULT hr;
	D3DXMATRIXA16 mWorldViewProj = m_mWorld;
    D3DXMatrixMultiply( &mWorldViewProj, &mWorldViewProj, &mLightViewProj );
    V( pEffect->SetMatrix( "g_mWorldViewProj", &mWorldViewProj ) );

	// ...Setup world-light view matrix
	D3DXMATRIX mWorldLightView;
	D3DXMatrixMultiply(&mWorldLightView, &m_mWorld, &mLightView);
	pEffect->SetMatrix("g_mWorldLightView", &mWorldLightView);

    //LPD3DXMESH pMesh = m_Mesh.GetMesh();
    UINT cPass;
    V( pEffect->Begin( &cPass, 0 ) );
    for( UINT p = 0; p < cPass; ++p )
    {
        V( pEffect->BeginPass( p ) );

        for( DWORD i = 0; i < m_Mesh.m_dwNumMaterials; ++i )
        {
            V( pEffect->CommitChanges() );
            V( m_Mesh.GetMesh()->DrawSubset( i ) );
        }
        V( pEffect->EndPass() );
    }
    V( pEffect->End() );
}

void CObj::PointLightRender(ID3DXEffect* pEffect, const D3DXMATRIX& cameraViewProj, const D3DXMATRIX& lightView)
{
	HRESULT hr = 0;
	// Setup the shader constants
	// ...Setup world-camera view-projection matrix
	D3DXMATRIX mWorldViewProj;	
	D3DXMatrixMultiply(&mWorldViewProj, &m_mWorld, &cameraViewProj);
	pEffect->SetMatrix("g_mWorldViewProj", &mWorldViewProj);
	// ...Setup world-light view matrix
	D3DXMATRIX mWorldLightView;
	D3DXMatrixMultiply(&mWorldLightView, &m_mWorld, &lightView);
	pEffect->SetMatrix("g_mWorldLightView", &mWorldLightView);

	LPD3DXMESH pMesh = m_Mesh.GetMesh();
    UINT cPass;
    V( pEffect->Begin( &cPass, 0 ) );
    for( UINT p = 0; p < cPass; ++p )
    {
        V( pEffect->BeginPass( p ) );

        for( DWORD i = 0; i < m_Mesh.m_dwNumMaterials; ++i )
        {
            if( m_Mesh.m_pTextures[i] )
                V( pEffect->SetTexture( "g_txDiffuse", m_Mesh.m_pTextures[i] ) )
            else
                V( pEffect->SetTexture( "g_txDiffuse", NULL ) )
            V( pEffect->CommitChanges() );
            V( pMesh->DrawSubset( i ) );
        }
        V( pEffect->EndPass() );
    }
    V( pEffect->End() );
}