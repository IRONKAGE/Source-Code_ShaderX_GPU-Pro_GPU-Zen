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
#include "RenderWorld.h"

#include "DXUT.h"
#include <boost/foreach.hpp>

#include "RenderSystem.h"

RenderWorld::RenderWorld (RenderSystem* r)
: renderSystem_ (r)
{
}

void RenderWorld::AddRenderEntity(std::tr1::shared_ptr<RenderEntity> entity)
{
	entities_.push_back (entity);
}

void RenderWorld::Render(const D3DXMATRIX &viewProjection)
{
	DefaultRenderAction dea;

	Render (viewProjection, &dea);
}

void RenderWorld::Render (const D3DXMATRIX& viewProjection, IRenderAction* a)
{
	renderSystem_->SetViewProjectionMatrix (viewProjection);

	a->PreRender (renderSystem_);

	BOOST_FOREACH(const std::tr1::shared_ptr<RenderEntity>& e, entities_)
	{
		a->Render (renderSystem_, *e);
	}

	a->PostRender (renderSystem_);
}

IRenderAction::~IRenderAction ()
{
}

void IRenderAction::PreRender (RenderSystem* r)
{
	PreRenderImpl (r);
}

void IRenderAction::Render (RenderSystem* r, RenderEntity& e)
{
	RenderImpl (r, e);
}

void IRenderAction::PostRender (RenderSystem* r)
{
	PostRenderImpl (r);
}

void DefaultRenderAction::PreRenderImpl (RenderSystem* r)
{
}

void DefaultRenderAction::RenderImpl (RenderSystem* r, RenderEntity& e)
{
	r->Render (e);
}

void DefaultRenderAction::PostRenderImpl (RenderSystem* r)
{
}