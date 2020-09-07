//--------------------------------------------------------------------------------------
// File: PointLight.h
//
// Encapsulates a point light in the scene by grouping its world matrix with the light
//	parameter.
//
// Copyright (c) Hung-Chien Liao. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once
#include "Sphere.h"

// Forward declarations
struct CObj;

class CPointLight
{
public:
	static const float TSMFaceFOVX;
	static const float TSMFaceFOVY;
	static const int ShadowMapSize;
	static const int CubeShadowMapSize;

	CObj*			m_pObj;
	D3DLIGHT9		m_Light;
	D3DXMATRIX		m_mWorld;
	CSphere			m_WorldBound;
	bool			m_bOn;

	LRESULT HandleMessages( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam );

	// Calculate the view projection matrix for each tetrahedron face
	void CalTSMViewProjMatrix();

	// Calculate the view projection matrix for each tetrahedron face with stencil
	void CalTSMStencilViewProjMatrix();

	// Calculate the view projection matrix for each cube face
	void CalCubeViewProjMatrix();

	// Generate the tetrahedron shadow map
	void GenTSM(IDirect3DDevice9* pd3dDevice, std::list<CObj*>& objsInLight);

	// Generate the tetrahedron shadow map with lookup texture
	void GenTSMLookUp(IDirect3DDevice9* pd3dDevice, std::list<CObj*>& objsInLight);

	// Generate the Dual-paraboloid shadow map
	void GenDSM(IDirect3DDevice9* pd3dDevice, std::list<CObj*>& objsInLight);

	// Generate the cube shadow map
	void GenCubeSM(IDirect3DDevice9* pd3dDevice, std::list<CObj*>& objsInLight);

	// Use this light to render the scene with Tetrahedron shadow map
	void RenderSceneTSM(IDirect3DDevice9* pd3dDevice, CObj* allObjs, unsigned int numObjs);

	// Use this light to render the scene with Tetrahedron shadow map and lookup texture
	void RenderSceneTSMLookUp(IDirect3DDevice9* pd3dDevice, CObj* allObjs, unsigned int numObjs);

	// Use this light to render the scene with Dual-paraboloid shadow map
	void RenderSceneDSM(IDirect3DDevice9* pd3dDevice, CObj* allObjs, unsigned int numObjs);

	// Use this light to render the scene with cube shadow map
	void RenderSceneCubeSM(IDirect3DDevice9* pd3dDevice, CObj* allObjs, unsigned int numObjs);

	// Render this light model
	void RenderLight(IDirect3DDevice9* pd3dDevice);

	// Use this function to create the lookup texture
	void CreateCubeToTSMCoord(IDirect3DDevice9* pd3dDevice);
};