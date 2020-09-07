#include "dxstdafx.h"
#include "Role.h"
#include "RenderContext.h"

Role::Role(LPD3DXBUFFER materialBuffer, unsigned int nSubmeshes, TextureDirectory& textureDirectory, LPDIRECT3DDEVICE9 device)
{
	D3DXMATERIAL* materialArray = (D3DXMATERIAL*)materialBuffer->GetBufferPointer();
	for(int t=0; t<nSubmeshes; t++)
	{
		LPDIRECT3DTEXTURE9 texture;
		TextureDirectory::iterator iTex = textureDirectory.find(materialArray[t].pTextureFilename);
		if(iTex == textureDirectory.end())
		{
			char textureFilePath[512];
			strcpy(textureFilePath, "media\\");
			strcat(textureFilePath, materialArray[t].pTextureFilename);
			HRESULT hr = D3DXCreateTextureFromFileA(device, textureFilePath, &texture);
			if(hr != S_OK)
				texture = NULL;
			textureDirectory[materialArray[t].pTextureFilename] = texture;
		}
		else
			texture = iTex->second;

		Material* material = new Material("defaultTechnique");
		material->setTexture("kdMap", texture);
		material->setVector("kdColor", D3DXVECTOR4(1, 1, 1, 1));
		materials.push_back(material);
	}
}

Role::Role()
{
}

Role::~Role(void)
{
	std::vector<Material*>::iterator i = materials.begin();
	while(i != materials.end())
	{
		delete *i;
		i++;
	}
}


void Role::render(const RenderContext& context, LPD3DXMESH mesh)
{
	unsigned int nSubmeshes = materials.size();
	for(unsigned int i=0; i<nSubmeshes; i++)
	{
		materials[i]->apply(context.effect);
		unsigned int nPasses = 0;
		context.effect->Begin(&nPasses, 0);
		for(unsigned int p=0; p<nPasses; p++)
		{
			context.effect->BeginPass(p);
			mesh->DrawSubset(i);
			context.effect->EndPass();
		}
		context.effect->End();
	}
}

void Role::addMaterial(Material* material)
{
	materials.push_back(material);
}
