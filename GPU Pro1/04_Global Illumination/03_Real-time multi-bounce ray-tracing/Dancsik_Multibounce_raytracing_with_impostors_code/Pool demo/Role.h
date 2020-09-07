#pragma once

#include "Directory.h"
#include "Material.h"
#include <vector>

class RenderContext;

class Role
{
	std::vector<Material*> materials;
public:
	Role(LPD3DXBUFFER materialBuffer, unsigned int nSubmeshes, TextureDirectory& textureDirectory, LPDIRECT3DDEVICE9 device);
	Role();
public:
	~Role(void);

	void render(const RenderContext& context, LPD3DXMESH mesh);

	void addMaterial(Material* material);
	Material* getMaterial(int i) {return materials[i];}
};
