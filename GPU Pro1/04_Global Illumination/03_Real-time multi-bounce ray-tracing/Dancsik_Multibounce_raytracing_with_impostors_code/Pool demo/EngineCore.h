#pragma once
#include "GameInterface.h"
#include "Directory.h"
#include "YawPitchRollCamera.h"

class XMLNode;
class NodeGroup;

class EngineCore : public GameInterface
{
protected:

	LPD3DXEFFECT effect;
	NodeGroup* worldRoot;

	ID3DXFont* Font;

	CFirstPersonCamera camera;

	MeshDirectory		meshDirectory;
	TextureDirectory	textureDirectory;
	CubeTextureDirectory cubeTextureDirectory;
	ShadedMeshDirectory	shadedMeshDirectory;
	EntityDirectory		entityDirectory;
	RigidModelDirectory	rigidModelDirectory;
	SpotLightDirectory spotLightDirectory;

	void loadLevel(wchar_t* xmlFileName);
	void loadGroup(XMLNode& groupNode, NodeGroup*& group);
	void loadRigidModels(XMLNode& xMainNode);
	void loadRigidBodies(XMLNode& groupNode, NodeGroup* group);
	void loadSpotlights(XMLNode& xMainNode);

	void renderHUD();

public:
	EngineCore(LPDIRECT3DDEVICE9 device);

	HRESULT createDefaultResources(wchar_t* effectFileName);
	HRESULT releaseDefaultResources();

	HRESULT createManagedResources();
	HRESULT releaseManagedResources();
			
	LPDIRECT3DCUBETEXTURE9 CreateCubeTexture(int size, D3DFORMAT Format);
	LPDIRECT3DTEXTURE9 CreateTexture(int width, int height, D3DFORMAT Format);
	LPDIRECT3DSURFACE9 CreateDepthStencilSurface(int width, int height);

	Entity* getFullScreenQuad();
	Entity* getEntity(wchar_t* entityName);
	SpotLight* getSpotLight(wchar_t* spotLightName);

	void setGlobalParameters(D3DXMATRIX viewMatrix, D3DXMATRIX projMatrix, D3DXVECTOR3 eyePosition);
	LPD3DXEFFECT getEffect() {return effect;}
	LPDIRECT3DDEVICE9 getDevice() {return device;}
	NodeGroup* getWorldRoot() {return worldRoot;}

	void processMessage( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	void animate(double dt, double t);
	void render();
};
