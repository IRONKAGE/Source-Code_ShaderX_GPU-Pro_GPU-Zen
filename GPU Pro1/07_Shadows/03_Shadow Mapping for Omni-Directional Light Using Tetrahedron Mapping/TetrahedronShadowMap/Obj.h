//--------------------------------------------------------------------------------------
// File: Obj.h
//
// Encapsulates a mesh object in the scene by grouping its world matrix with the mesh.
//
// Copyright (c) Hung-Chien Liao. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once
#include "SDKmesh.h"
#include "Sphere.h"

struct CObj
{
    CDXUTXFileMesh	m_Mesh;
    D3DXMATRIX		m_mWorld;
	CSphere			m_WorldBound;

	// render this object into shadow map
	void GenShadowMap(ID3DXEffect* pEffect, const D3DXMATRIXA16& mLightViewProj);

	// render this object into Dual-paraboloid shadow map
	void GenDSM(ID3DXEffect* pEffect, const D3DXMATRIX& mLightView);

	// render this object into cube shadow map
	void GenCubeShadowMap(ID3DXEffect* pEffect, const D3DXMATRIXA16& mLightViewProj, const D3DXMATRIXA16& mLightView);

	// Use point light to render
	void PointLightRender(ID3DXEffect* pEffect, const D3DXMATRIX& cameraViewProj, const D3DXMATRIX& lightView);
};