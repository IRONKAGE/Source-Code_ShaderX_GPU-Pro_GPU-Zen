#pragma once
#include "entity.h"

 

class EnvironmentMappedEntity : public Entity
{
protected:
	
	bool disableForces, updateFresnelWater, updateFresnelGlass, updateFresnelRefractor, updateRefractorPosition, updateRefractorIOR;

	EngineCore* core;
	
	//These matrices are responsible for the transformations between world space and the coordinate system of the waters height map.
	D3DXMATRIXA16 TBN, TBNInverse;
	
	float time;

	float snippetSize;			//size of the quadrilateral used for photon splatting
	unsigned int photonMapSize;		

	float fresnelFactorWater, fresnelFactorRefractor, fresnelFactorGlass;	//fresnel factors for different refractive and/or reflective elements
	float indexOfRefractionRefractor;		//IOR of the floating refractive object

	Entity* refractor;				//the floating refractive object
	SpotLight* causticLight;		//pointer to the light source
	Entity* fullScreenQuad;

	D3DXVECTOR3 environmentMapPosition;		//the position of the reference point belonging to the environment impostor in object space 

	LPDIRECT3DSURFACE9 environmentSurface;
	LPDIRECT3DCUBETEXTURE9 normalMapGlass;		//the object distance impostor of the container
	LPDIRECT3DCUBETEXTURE9 uvMap;				//the distance impostor of the diffuse environment

	LPDIRECT3DSURFACE9 heightMapSurface;		
	LPDIRECT3DTEXTURE9 heightMap;				//the height map of the water surface

	LPDIRECT3DSURFACE9 photonMapSurface;
	LPDIRECT3DTEXTURE9 photonMap;				//photon map. Photon hits are stored in texture space.

	LPDIRECT3DTEXTURE9 lightMap;				//light map
	LPDIRECT3DSURFACE9 lightMapStencilSurface;

	LPDIRECT3DTEXTURE9 gaussianFilterTexture;		//filter texture used by the photon splatting

	LPDIRECT3DVERTEXBUFFER9 causticQuadVertexBuffer;
	LPDIRECT3DINDEXBUFFER9 causticQuadIndexBuffer;
	LPDIRECT3DVERTEXBUFFER9 causticQuadInstanceBuffer;
	LPDIRECT3DVERTEXDECLARATION9 causticQuadVertexDecl;

	enum { RENDER_ENVIRONMENT_UV_DISTANCE, RENDER_GLASS_NORMAL_DISTANCE};

	D3DXVECTOR4 computeBoundingSphere(Entity* object1, Entity* object2);	
	void RenderSceneIntoCubeMap(int mode);		//creates the environment impostor and the containers object distance impostor 
	void renderHeightMap();		//renders the height map of the water surface
	void renderCaustics();		//generates the photon map and reders caustics to the light map
	void createCausticQuadrilaterals();
	//returns the normal vector of the water surface and the height of the water volume at a given location 
	float getWaterHeightAndNormal(D3DXVECTOR3 location, D3DXVECTOR3& normal);	


public:
	EnvironmentMappedEntity(ShadedMesh* shadedMesh);

	~EnvironmentMappedEntity(void);

	void setEnvironmentMapPosition(const D3DXVECTOR3& position);
	void addRefractor(Entity* refr) {refractor = refr;}
	void addCausticLight(SpotLight* light) {causticLight = light;}

	HRESULT createDefaultResources(EngineCore* core);
	HRESULT releaseDefaultResources();

	D3DXVECTOR3 getReferencePointPosition();
	float getFresnelFactorWater() {return fresnelFactorWater;}
	float getFresnelFactorRefractor() {return fresnelFactorRefractor;}
	float  getFresnelFactorGlass() {return fresnelFactorGlass;}
	float getIndexOfRefractionRefractor() {return indexOfRefractionRefractor;}
	bool areForcesDisabled() {return disableForces;}

	void handleMessage(HWND hWnd,  UINT uMsg, WPARAM wParam, LPARAM lParam);
	void animate(double dt);
	void control(double dt, Node* others);
};
