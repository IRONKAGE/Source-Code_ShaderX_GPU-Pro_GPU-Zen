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
#ifndef VTM_TILEID_MATERIAL_H
#define VTM_TILEID_MATERIAL_H

#include "Material.h"

struct ID3D10EffectScalarVariable;
struct ID3D10EffectVectorVariable;

struct TextureInfo_t;

struct VirtualTextureSettings_t
{
	int pageSize [2];
	int maximumMipMapLevel;
	int mipMapScaleFactor;
};

class TileIDMaterial : public Material
{
public:
	TileIDMaterial (ID3D10Device* d, const VirtualTextureSettings_t& settings);

	void SetTextureInfo (const TextureInfo_t& t);

private:
	void Setup (ID3D10Device* d);

	int tileSize_ [2];
	ID3D10EffectScalarVariable*	textureIdVariable_;
	ID3D10EffectVectorVariable* textureSizeVariable_;
	ID3D10EffectScalarVariable* textureMaxMipVariable_;
	ID3D10EffectVectorVariable* textureTileCount_;
	ID3D10EffectScalarVariable* scaleFactor_;
};

#endif