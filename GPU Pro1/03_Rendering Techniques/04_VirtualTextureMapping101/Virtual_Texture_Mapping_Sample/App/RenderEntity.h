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
#ifndef VTM_RENDER_ENTITY_H
#define VTM_RENDER_ENTITY_H

#include <D3DX10.h>

class RenderSystem;
struct Geometry_t;
struct RenderGeometry_t;
class Material;

class RenderEntity
{
public:
	RenderEntity (const Geometry_t& g, Material* m = 0);
	~RenderEntity ();

	void Init (RenderSystem* d);
	void Shutdown ();

	void SetWorldMatrix (D3DXMATRIX world);
	const D3DXMATRIX& GetWorldMatrix () const;

	void Setup (RenderSystem* d) const;
	void Render (RenderSystem* d) const;

	/**
	* Can be 0 if no material has been set
	*/
	Material*	GetMaterial () const
	{
		return material_;
	}

private:
	Geometry_t*			geometry_;
	RenderGeometry_t*	renderGeometry_;
	Material*			material_;

	D3DXMATRIX			world_;
};

#endif