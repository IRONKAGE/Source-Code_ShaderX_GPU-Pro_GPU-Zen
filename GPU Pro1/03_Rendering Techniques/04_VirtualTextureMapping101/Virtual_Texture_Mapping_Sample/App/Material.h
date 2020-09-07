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
#ifndef VTM_MATERIAL_H
#define VTM_MATERIAL_H

struct ID3D10Device;
struct D3DXMATRIX;
struct ID3D10Effect;
struct ID3D10EffectMatrixVariable;

struct TextureInfo_t;

class Material
{
public:
	Material (ID3D10Device* device, const wchar_t* filename);

	virtual ~Material();

	void SetWorldViewProjectionMatrix (const D3DXMATRIX& worldViewProjection);

	void Bind (ID3D10Device* device);

	const TextureInfo_t&	GetTextureInfo () const;
protected:
	ID3D10Effect* effect_;

private:
	virtual void Setup (ID3D10Device* d);
	virtual const TextureInfo_t& GetTextureInfoImpl () const;
	ID3D10EffectMatrixVariable* worldViewProjectionVariable_;
};	


#endif