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
#include "DXUT.h"

#include "TerrainTexturedMaterial.h"

TerrainTexturedMaterial::TerrainTexturedMaterial (ID3D10Device* d,
												  ID3D10Texture2D* indirection, ID3D10Texture2D* pages)
												  : Material (d, TEXT("Shaders\\terrain.fx")), textureInfo_ (1, 8192, 8192)
{
	d->CreateShaderResourceView (indirection, NULL, &indirectionView_);
	d->CreateShaderResourceView (pages, NULL, &pageView_);

	pageViewVariable_ = effect_->GetVariableByName ("Cache_Texture")->AsShaderResource ();
	indirectionViewVariable_ = effect_->GetVariableByName ("Cache_Indirection")->AsShaderResource ();

	pageViewVariable_->SetResource (pageView_);
	indirectionViewVariable_->SetResource (indirectionView_);

	virtualTextureSizeVariable_ = effect_->GetVariableByName ("Virtual_TextureSize")->AsVector ();
	physicalPageSizeVariable_ = effect_->GetVariableByName ("Cache_PageSize")->AsVector ();
	physicalSizeVariable_ = effect_->GetVariableByName ("Cache_Size")->AsVector ();
	virtualTextureMaxMipMapLevel_ = effect_->GetVariableByName ("Virtual_MaxMipMapLevel")->AsScalar ();

	float c [] = { 8192, 8192 };
	virtualTextureSizeVariable_->SetFloatVector (c);
	virtualTextureMaxMipMapLevel_->SetFloat (7);
	float s [] = { 2048, 2048 };
	physicalSizeVariable_->SetFloatVector (s);
}

void	TerrainTexturedMaterial::SetPhysicalPageSize (float s)
{
	float size[] = { s, s};
	physicalPageSizeVariable_->SetFloatVector (size);
}

TerrainTexturedMaterial::~TerrainTexturedMaterial ()
{
	SAFE_RELEASE(indirectionView_);
	SAFE_RELEASE(pageView_);
}

void TerrainTexturedMaterial::Setup (ID3D10Device* d)
{
}

const TextureInfo_t& TerrainTexturedMaterial::GetTextureInfoImpl () const
{
	return textureInfo_;
}