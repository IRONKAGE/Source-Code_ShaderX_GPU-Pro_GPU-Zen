#pragma once
#include "ShadedMesh.h"

class RefractorMesh : public ShadedMesh
{
protected:

	LPDIRECT3DSURFACE9 refractorSurface;
	LPDIRECT3DCUBETEXTURE9 refractorMap;		//object distance impostor. It stores surface normals and distances from the object centre

	int geometryImageSize;

	//the geometry image stores world space position of the surface points and normals (the latter are stored in the texture called 'normalMap')
	LPDIRECT3DTEXTURE9 geometryImage;			 		
	LPDIRECT3DSURFACE9 geometryImageStencilSurface;
	LPDIRECT3DTEXTURE9 normalMap;				

	//the min-max maps define a bounding box hierarchy,
	//The traversal order of the hierarchy is described by pointers stored in the link map  
	LPDIRECT3DTEXTURE9 minMap, maxMap, linkMap;
	LPDIRECT3DSURFACE9 minMaxStencilSurface;


	bool RenderSceneIntoCubeMap(EngineCore* core);	//renders the surface normals in an object distance impostor

	void renderGeometryImage(EngineCore* core);		//renders the geometry image
	void renderMinMaxMaps(EngineCore* core);		//renders the min map and the max map
	void renderLinkMap(EngineCore* core);			//renders the link map
	
public:
	RefractorMesh(LPD3DXMESH mesh);
	RefractorMesh(LPD3DXMESH mesh, LPD3DXBUFFER materialBuffer, unsigned int nSubMeshes, TextureDirectory& textureDirectory, LPDIRECT3DDEVICE9 device);

	HRESULT createDefaultResources(EngineCore* core);
	HRESULT releaseDefaultResources();
	LPDIRECT3DCUBETEXTURE9 getRefractorMap() {return refractorMap;}
};
