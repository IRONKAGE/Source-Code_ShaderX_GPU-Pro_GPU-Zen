#pragma once

//#include "Directory.h"
#include "Role.h"
#include "Entity.h"

class ShadedMesh
{
protected:
	LPD3DXMESH mesh;
	RoleDirectory roles;
public:
	D3DXVECTOR3 boundingBoxMin;
	D3DXVECTOR3 boundingBoxMax;

	ShadedMesh(LPD3DXMESH mesh, LPD3DXBUFFER materialBuffer, unsigned int nSubMeshes, TextureDirectory& textureDirectory, LPDIRECT3DDEVICE9 device);
	ShadedMesh(LPD3DXMESH mesh);
	~ShadedMesh(void);

	void computeBoundingBox();

	void addRole(const UnicodeString& name, Role* role);
	Role* getRole(const UnicodeString& s) {return roles[s];}

	virtual HRESULT createDefaultResources(EngineCore* core) {return S_OK;}
	virtual HRESULT releaseDefaultResources() {return S_OK;}
	virtual void render(const RenderContext& context);
};
