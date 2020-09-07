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
#include "TileIdRenderAction.h"

#include "RenderEntity.h"
#include "RenderSystem.h"
#include "TextureInfo.h"
#include "TileIDMaterial.h"

TileIDRenderAction::TileIDRenderAction (RenderTarget_t* myTarget, TileIDMaterial* material)
: rt_ (myTarget), m_ (material)
{
}

void TileIDRenderAction::PreRenderImpl (RenderSystem* r)
{
	r->SetRenderTarget (*rt_);
	const float clearColor [] = { 0.0f, 0.0f, 0.0f, 0.0f };
	r->Clear (*rt_, clearColor, 1.0f);
}

void TileIDRenderAction::RenderImpl (RenderSystem* r, RenderEntity& e)
{
	if (e.GetMaterial ()->GetTextureInfo ().IsValid ())
	{
		const TextureInfo_t& ti = e.GetMaterial ()->GetTextureInfo ();

		m_->SetTextureInfo (ti);
	}

	r->Render (e, *m_);
}

void TileIDRenderAction::PostRenderImpl (RenderSystem* r)
{
	r->SetDefaultRenderTarget ();
}