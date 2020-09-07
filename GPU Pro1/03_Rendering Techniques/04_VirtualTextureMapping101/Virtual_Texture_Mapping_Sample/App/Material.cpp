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

#include "Material.h"

#include <string>
#include "TextureInfo.h"

/////////////////////////////////////////////////////////////////////////////
Material::Material (ID3D10Device* device, const wchar_t* filename)
{
	ID3D10Blob* blob = 0;
	HRESULT r = D3DX10CreateEffectFromFile (filename,
		NULL,
		NULL,
		"fx_4_0",
		D3D10_SHADER_DEBUG | /* D3D10_SHADER_ENABLE_STRICTNESS | */ D3D10_SHADER_SKIP_OPTIMIZATION, 
		0,
		device,
		NULL,
		NULL,
		&effect_,
		&blob,
		NULL);

	if (FAILED(r) && blob)
	{
		std::string m ((char*)blob->GetBufferPointer (), blob->GetBufferSize ());
		__debugbreak ();
		blob->Release ();
	}

	worldViewProjectionVariable_ = 
		effect_->GetVariableBySemantic ("WorldViewProjection")->AsMatrix ();
}

/////////////////////////////////////////////////////////////////////////////
Material::~Material()
{
	SAFE_RELEASE(effect_);
}

/////////////////////////////////////////////////////////////////////////////
void Material::SetWorldViewProjectionMatrix (const D3DXMATRIX& worldViewProjection)
{
	worldViewProjectionVariable_->SetMatrix (	// This should not modify the matrix
		const_cast<float*> (static_cast<const float*> (worldViewProjection)));
}

/////////////////////////////////////////////////////////////////////////////
void Material::Bind (ID3D10Device* device)
{
	Setup (device);
	effect_->GetTechniqueByName ("Render")->GetPassByIndex (0)->Apply (0);
}

/////////////////////////////////////////////////////////////////////////////
void Material::Setup (ID3D10Device* /* d */)
{
}

/////////////////////////////////////////////////////////////////////////////
const TextureInfo_t& Material::GetTextureInfo () const
{
	return GetTextureInfoImpl ();
}

namespace
{
	TextureInfo_t invalid;
}

/////////////////////////////////////////////////////////////////////////////
const TextureInfo_t& Material::GetTextureInfoImpl () const
{
	return invalid;
}