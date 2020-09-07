#include "dxstdafx.h"
#include "Material.h"

Material::Material(const AnsiString& techniqueName)
:techniqueName(techniqueName)
{
}

Material::~Material(void)
{
}

void Material::setTexture(const AnsiString& textureName, LPDIRECT3DTEXTURE9 texture)
{
	textureEffectVariables[textureName] = texture;
}

void Material::setCubeTexture(const AnsiString& cubeTextureName, LPDIRECT3DCUBETEXTURE9 cubeTexture)
{
	cubeTextureEffectVariables[cubeTextureName] = cubeTexture;
}

void Material::setVector(const AnsiString& vectorName, const D3DXVECTOR4& value)
{
	vectorEffectVariables[vectorName] = value;
}


void Material::apply(LPD3DXEFFECT effect)
{
	effect->SetTechnique(techniqueName);
	{
		TextureDirectory::iterator i = textureEffectVariables.begin();
		while(i != textureEffectVariables.end())
		{
			effect->SetTexture(i->first, i->second);
			i++;
		}
	}
	{
		CubeTextureDirectory::iterator i = cubeTextureEffectVariables.begin();
		while(i != cubeTextureEffectVariables.end())
		{
			effect->SetTexture(i->first, i->second);
			i++;
		}
	}
	{
		VectorDirectory::iterator i = vectorEffectVariables.begin();
		while(i != vectorEffectVariables.end())
		{
			effect->SetVector(i->first, &i->second);
			i++;
		}
	}
}