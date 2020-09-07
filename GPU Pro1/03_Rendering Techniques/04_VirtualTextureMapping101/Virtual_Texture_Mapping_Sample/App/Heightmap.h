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
#ifndef VTM_HEIGHTMAP_H
#define VTM_HEIGHTMAP_H

#include <D3DX10.h>
#include <D3DX10Math.h>
#include <boost/cstdint.hpp>

#include <boost/tr1/memory.hpp>

class RenderEntity;
class RenderSystem;
class Material;

class Heightmap
{
public:
	Heightmap (RenderSystem* r, wchar_t* filename, ID3D10Texture2D* pages, ID3D10Texture2D* indirection,
		float pageSize);

	~Heightmap ();

	const std::tr1::shared_ptr<RenderEntity>&	GetRenderEntity ()
	{
		return renderEntity_;
	}

private:
	boost::uint8_t* data_;

	std::tr1::shared_ptr<RenderEntity>	renderEntity_;
	Material*							material_;

	ID3D10EffectMatrixVariable* matrixWorld_;
	ID3D10EffectMatrixVariable* matrixWorldViewProjection_;
};

#endif