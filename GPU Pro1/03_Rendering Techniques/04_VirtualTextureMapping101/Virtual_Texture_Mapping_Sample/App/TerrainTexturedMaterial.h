/*
	Virtual texture mapping demo app
    Copyright (C) 2008, 2009 Matthäus G. Chajdas
    Contact: shaderx8@anteru.net

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef VTM_TERRAIN_TEXTURED_MATERIAL_H
#define VTM_TERRAIN_TEXTURED_MATERIAL_H

#include "Material.h"
#include "TextureInfo.h"

class TerrainTexturedMaterial : public Material
{
public:
	TerrainTexturedMaterial (ID3D10Device* d,
		ID3D10Texture2D* indirection, ID3D10Texture2D* pages);
	~TerrainTexturedMaterial ();

	void	SetPhysicalPageSize (float s);

private:
	void Setup (ID3D10Device* d);
	const TextureInfo_t& GetTextureInfoImpl () const;

	TextureInfo_t						textureInfo_;

	ID3D10ShaderResourceView*			indirectionView_;
	ID3D10ShaderResourceView*			pageView_;

	ID3D10EffectShaderResourceVariable* indirectionViewVariable_;
	ID3D10EffectShaderResourceVariable* pageViewVariable_;
	ID3D10EffectVectorVariable*			virtualTextureSizeVariable_;
	ID3D10EffectVectorVariable*			physicalPageSizeVariable_;
	ID3D10EffectVectorVariable*			physicalSizeVariable_;
	ID3D10EffectScalarVariable*			virtualTextureMaxMipMapLevel_;
};

#endif