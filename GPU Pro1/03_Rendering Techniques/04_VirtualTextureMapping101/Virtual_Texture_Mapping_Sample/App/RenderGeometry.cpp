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

#include "RenderGeometry.h"

#include <boost/foreach.hpp>

#include "Geometry.h"

RenderGeometry_t::RenderGeometry_t (
		ID3D10Device* d,
		Geometry_t* geometry,
		ID3D10InputLayout* layout)
		: inputLayout_ (layout), type_ (geometry->type)
	{
		// Create the index buffer
		D3D10_BUFFER_DESC indexBufferDesc;
		::ZeroMemory (&indexBufferDesc, sizeof (indexBufferDesc));

		indexBufferDesc.BindFlags = D3D10_BIND_INDEX_BUFFER;
		indexBufferDesc.ByteWidth = sizeof (int) * geometry->indices.size ();
		indexBufferDesc.Usage = D3D10_USAGE_IMMUTABLE;

		D3D10_SUBRESOURCE_DATA indexBufferData;
		::ZeroMemory (&indexBufferData, sizeof (indexBufferData));
		indexBufferData.pSysMem = &(geometry->indices [0]);

		d->CreateBuffer (&indexBufferDesc, &indexBufferData, &indexBuffer_);

		BOOST_FOREACH(const VertexBufferData_t& data, geometry->vertices)
		{
			AddVertexBuffer (d, data);
		}

		indexCount_ = geometry->indices.size ();
	}

	void RenderGeometry_t::AddVertexBuffer (ID3D10Device* d, const VertexBufferData_t& data)
	{
		D3D10_BUFFER_DESC vertexBufferDesc;
		::ZeroMemory (&vertexBufferDesc, sizeof (vertexBufferDesc));
		vertexBufferDesc.ByteWidth = data.data.size ();
		vertexBufferDesc.Usage = D3D10_USAGE_IMMUTABLE;
		vertexBufferDesc.BindFlags = D3D10_BIND_VERTEX_BUFFER;

		D3D10_SUBRESOURCE_DATA vertexBufferData;
		::ZeroMemory (&vertexBufferData, sizeof (vertexBufferData));
		vertexBufferData.pSysMem = &data.data [0];

		ID3D10Buffer* b;
		d->CreateBuffer (&vertexBufferDesc, &vertexBufferData, &b);

		vertexBuffers_.push_back (b);
		strides_.push_back (data.elementSize);
		offsets_.push_back (0);
	}

	RenderGeometry_t::~RenderGeometry_t ()
	{
		SAFE_RELEASE(indexBuffer_);

		for (size_t i = 0; i < vertexBuffers_.size (); ++i)
		{
			SAFE_RELEASE(vertexBuffers_ [i]);
		}
	}

	void RenderGeometry_t::Setup (ID3D10Device* d)
	{
		d->IASetInputLayout (inputLayout_);
		d->IASetVertexBuffers (0, vertexBuffers_.size (), &vertexBuffers_ [0], &strides_[0], &offsets_ [0]);
		d->IASetIndexBuffer (indexBuffer_, DXGI_FORMAT_R32_UINT, 0);
		d->IASetPrimitiveTopology (type_);
	}

	void RenderGeometry_t::Render (ID3D10Device* d)
	{
		d->DrawIndexed (indexCount_, 0, 0);
	}