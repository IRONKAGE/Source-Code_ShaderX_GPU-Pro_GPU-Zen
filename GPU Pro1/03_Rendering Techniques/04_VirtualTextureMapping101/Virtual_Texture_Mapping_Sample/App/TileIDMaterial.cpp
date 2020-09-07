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
#include "TileIDMaterial.h"

#include <D3D10.h>
#undef max
#include <algorithm>
#include "TextureInfo.h"

/////////////////////////////////////////////////////////////////////////////
TileIDMaterial::TileIDMaterial (ID3D10Device* d, const VirtualTextureSettings_t& settings)
: Material (d, TEXT("Shaders\\TileID2.fx"))
{
	tileSize_ [0] = settings.pageSize [0];
	tileSize_ [1] = settings.pageSize [1];

	textureIdVariable_ = effect_->GetVariableByName ("TextureID")->AsScalar ();
	textureSizeVariable_ = effect_->GetVariableByName ("TextureSize")->AsVector ();
	textureMaxMipVariable_ = effect_->GetVariableByName ("MaximumMipMapLevel")->AsScalar ();
	textureTileCount_ = effect_->GetVariableByName ("TileCount")->AsVector ();
	scaleFactor_ = effect_->GetVariableByName ("MipMapScaleFactor")->AsScalar ();

	textureMaxMipVariable_->SetFloat (
		static_cast<float> (settings.maximumMipMapLevel)
		);

	scaleFactor_->SetFloat (
		static_cast<float> (settings.mipMapScaleFactor)
		);
}

/////////////////////////////////////////////////////////////////////////////
void TileIDMaterial::Setup (ID3D10Device*)
{	
}

/////////////////////////////////////////////////////////////////////////////
void TileIDMaterial::SetTextureInfo (const TextureInfo_t& ti)
{
	float s [] = 
	{ 
		static_cast<float> (ti.width / tileSize_ [0]),
		static_cast<float> (ti.height / tileSize_ [1]) 
	};

	float t [] = 
	{
		static_cast<float> (ti.width),
		static_cast<float> (ti.height)
	};

	textureTileCount_->SetFloatVector (s);
	textureSizeVariable_->SetFloatVector (t);

	textureIdVariable_->SetFloat (static_cast<float> (ti.id));
}
