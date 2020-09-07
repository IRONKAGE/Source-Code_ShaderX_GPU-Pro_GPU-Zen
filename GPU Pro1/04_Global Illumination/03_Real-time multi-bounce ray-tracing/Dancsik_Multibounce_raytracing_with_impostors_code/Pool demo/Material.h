#pragma once
#include "Directory.h"
#include "AnsiString.h"

class Material
{
	TextureDirectory textureEffectVariables;
	CubeTextureDirectory cubeTextureEffectVariables;
	VectorDirectory vectorEffectVariables;

	const AnsiString techniqueName;
public:
	Material(const AnsiString& techniqueName);
public:
	~Material(void);

	void setTexture(const AnsiString& textureName, LPDIRECT3DTEXTURE9 texture);
	void setCubeTexture(const AnsiString& cubeTextureName, LPDIRECT3DCUBETEXTURE9 cubeTexture);
	void setVector(const AnsiString& vectorName, const D3DXVECTOR4& value);

	LPDIRECT3DTEXTURE9 getTexture(const AnsiString& textureName) {return textureEffectVariables[textureName];}
	D3DXVECTOR4 getVector(const AnsiString& vectorName) {return vectorEffectVariables[vectorName];}

	void apply(LPD3DXEFFECT effect);
};
