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

#include "RenderEntity.h"

#include "RenderGeometry.h"
#include "Geometry.h"
#include "RenderSystem.h"

/////////////////////////////////////////////////////////////////////////////
RenderEntity::RenderEntity (const Geometry_t& g, Material* m)
: geometry_ (new Geometry_t (g)), material_ (m)
{			
	D3DXMatrixIdentity (&world_);
}

/////////////////////////////////////////////////////////////////////////////
RenderEntity::~RenderEntity ()
{
	delete geometry_;
}

/////////////////////////////////////////////////////////////////////////////
void RenderEntity::Init (RenderSystem* d)
{
	renderGeometry_ = new RenderGeometry_t (d->GetDevice (), geometry_, d->GetDefaultLayout ());
}

/////////////////////////////////////////////////////////////////////////////
void RenderEntity::Shutdown ()
{
	delete renderGeometry_;
}

/////////////////////////////////////////////////////////////////////////////
void RenderEntity::SetWorldMatrix (D3DXMATRIX world)
{
	world_ = world;
}

/////////////////////////////////////////////////////////////////////////////
const D3DXMATRIX& RenderEntity::GetWorldMatrix () const
{
	return world_;
}

/////////////////////////////////////////////////////////////////////////////
void RenderEntity::Setup (RenderSystem* d) const
{
	renderGeometry_->Setup (d->GetDevice ());
}

/////////////////////////////////////////////////////////////////////////////
void RenderEntity::Render (RenderSystem* d) const
{
	renderGeometry_->Render (d->GetDevice ());
}