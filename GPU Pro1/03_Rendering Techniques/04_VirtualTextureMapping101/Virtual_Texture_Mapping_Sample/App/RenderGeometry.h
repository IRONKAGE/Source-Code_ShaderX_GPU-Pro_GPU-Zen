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
#ifndef VTM_RENDER_GEOMETRY_H
#define VTM_RENDER_GEOMETRY_H

#include <vector>

struct Geometry_t;
struct VertexBufferData_t;

struct RenderGeometry_t
{
public:
	RenderGeometry_t (
		ID3D10Device* d,
		Geometry_t* geometry,
		ID3D10InputLayout* layout);

	~RenderGeometry_t ();

	void Setup (ID3D10Device* d);
	void Render (ID3D10Device* d);

private:
	void AddVertexBuffer (ID3D10Device* d, const VertexBufferData_t& data);

	std::vector<ID3D10Buffer*>			vertexBuffers_;
	std::vector<UINT>					strides_;
	std::vector<UINT>					offsets_;

	ID3D10Buffer*						indexBuffer_;
	ID3D10InputLayout*					inputLayout_;

	int									indexCount_;

	D3D10_PRIMITIVE_TOPOLOGY			type_;
};

#endif