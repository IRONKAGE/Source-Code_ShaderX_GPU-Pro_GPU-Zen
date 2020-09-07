/*
	Virtual texture mapping demo app
    Copyright (C) 2008, 2009 Matth�us G. Chajdas
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
#ifndef VTM_TILE_ID_RENDER_ACTION_H
#define VTM_TILE_ID_RENDER_ACTION_H

#include "RenderWorld.h"

struct RenderTarget_t;
class TileIDMaterial;
class RenderSystem;

class TileIDRenderAction : public IRenderAction
{
public:
	TileIDRenderAction (RenderTarget_t* myTarget, TileIDMaterial* material);

	void PreRenderImpl (RenderSystem* r);

	void RenderImpl (RenderSystem* r, RenderEntity& e);

	void PostRenderImpl (RenderSystem* r);

private:
	RenderTarget_t* rt_;
	TileIDMaterial* m_;
};

#endif