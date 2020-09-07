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

#include "Heightmap.h"

#include <algorithm>
#include <vector>

#include "RenderEntity.h"
#include "RenderSystem.h"
#include "Geometry.h"
#include "TerrainTexturedMaterial.h"

/////////////////////////////////////////////////////////////////////////////
Heightmap::Heightmap (RenderSystem* r, wchar_t* filename, ID3D10Texture2D* pages, ID3D10Texture2D* indirection,
					  float pageSize)
{
	D3DX10_IMAGE_INFO info;
	D3DX10_IMAGE_LOAD_INFO linfo;
	::ZeroMemory (&linfo, sizeof (linfo));

	D3DX10GetImageInfoFromFile(TEXT("Textures\\height.bmp"), NULL, &info, NULL);

	static const int gridresolution = 256;

	linfo.Width = gridresolution;
	linfo.Height = gridresolution;
	linfo.Usage = D3D10_USAGE_STAGING;
	linfo.Format = DXGI_FORMAT_R8_UNORM;
	linfo.CpuAccessFlags = D3D10_CPU_ACCESS_READ;
	linfo.pSrcInfo = &info;

	const int imageSize = gridresolution * gridresolution;

	data_ = new boost::uint8_t [imageSize];

	ID3D10Texture2D* texture;
	D3DX10CreateTextureFromFile (r->GetDevice (), filename, &linfo, NULL, (ID3D10Resource**)&texture, NULL);

	// Copy the texture file into a byte array
	D3D10_MAPPED_TEXTURE2D map;
	texture->Map (D3D10CalcSubresource (0, 0, 0), D3D10_MAP_READ, 0, &map);

	std::copy (static_cast<boost::uint8_t*>(map.pData),
		static_cast<boost::uint8_t*>(map.pData) + imageSize,
		data_);

	texture->Unmap (D3D10CalcSubresource (0, 0, 0));

	// Discard the texture again
	texture->Release ();

	// Create the grid now
	std::vector<D3DXVECTOR3> grid;
	std::vector<D3DXVECTOR2> uvs;
	std::vector<D3DXVECTOR3> normals;
	grid.resize (imageSize);
	uvs.resize (imageSize);
	normals.resize (imageSize);

	for (int y = 0; y < gridresolution; ++y)
	{
		for (int x = 0; x < gridresolution; ++x)
		{
			grid [y * gridresolution + x] = D3DXVECTOR3 (
				x / static_cast<float> (gridresolution) * 64.0f,
				data_ [y * gridresolution + x] / 255.0f * 8,
				y / static_cast<float> (gridresolution) * 64.0f);

			uvs [y * gridresolution + x] = D3DXVECTOR2 (
				1 - x / static_cast<float> (gridresolution),
				y / static_cast<float> (gridresolution));
		}
	}

	for (int y = 0; y < gridresolution; ++y)
	{
		for (int x = 0; x < gridresolution; ++x)
		{
			const int offset = y * gridresolution + x;

			// Compute the derivatives to obtain the normal
			int nX1 = offset;
			int nX2 = offset;
			int pX1 = offset;
			int pX2 = offset;

			int nZ1 = offset;
			int nZ2 = offset;
			int pZ1 = offset;
			int pZ2 = offset;

			if (x < (gridresolution - 1)) { ++ nX1; ++ nX2; }
			if (x < (gridresolution - 2)) { ++ nX2; }
			if (x > 1) { -- pX1; -- pX2; }
			if (x > 2) { -- pX2; }

			if (y < (gridresolution - 1)) { nZ1 += gridresolution; nZ2 += gridresolution; }
			if (y < (gridresolution - 2)) { nZ2 += gridresolution; }
			if (y > 1) { pZ1 -= gridresolution; pZ2 -= gridresolution; }
			if (y > 2) { pZ2 -= gridresolution; }

			const D3DXVECTOR3 dX = 
				(- grid [nX2] + 8 * grid [nX1] - 8 * grid [pX1] + grid [pX2]);

			const D3DXVECTOR3 dZ = 
				(- grid [nZ2] + 8 * grid [nZ1] - 8 * grid [pZ1] + grid [pZ2]);

			D3DXVECTOR3 grad;
			D3DXVec3Normalize (& normals [offset], D3DXVec3Cross (&grad, &dZ, &dX));
		}
	}

	// Continue with the index buffer
	const int skipRow = gridresolution;

	int row = 0;

	Geometry_t g;
	g.type = D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

	for (int j = 0; j < gridresolution - 1; ++ j)
	{
		for (int i = 0; i < gridresolution - 1; ++i)
		{
			g.indices.push_back (row + i);
			g.indices.push_back (row + i + skipRow);
			g.indices.push_back (row + i + 1);

			g.indices.push_back (row + i + 1);
			g.indices.push_back (row + i + skipRow);
			g.indices.push_back (row + i + skipRow + 1);
		}

		row += skipRow;
	}	
	
	g.vertices.push_back (VertexBufferData_t (&grid[0],
		static_cast<int> (sizeof (D3DXVECTOR3) * grid.size ()),
		sizeof (D3DXVECTOR3)));

	g.vertices.push_back (VertexBufferData_t (&uvs[0], 
		static_cast<int> (sizeof (D3DXVECTOR2) * uvs.size ()),
		sizeof (D3DXVECTOR2)));

	g.vertices.push_back (VertexBufferData_t (&normals [0],
		static_cast<int> (sizeof (D3DXVECTOR3) * normals.size ()),
		sizeof (D3DXVECTOR3)));

	TerrainTexturedMaterial* m = new TerrainTexturedMaterial (r->GetDevice (), indirection, pages);
	m->SetPhysicalPageSize (pageSize);
	material_ = m;

	renderEntity_ = std::tr1::shared_ptr<RenderEntity>
		(new RenderEntity (g, material_));

	renderEntity_->Init (r);
}

/////////////////////////////////////////////////////////////////////////////
Heightmap::~Heightmap ()
{
	renderEntity_->Shutdown ();

	delete material_;
	delete [] data_;
}