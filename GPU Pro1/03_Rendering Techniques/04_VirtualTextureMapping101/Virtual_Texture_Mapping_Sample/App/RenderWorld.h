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
#ifndef VTM_RENDER_WORLD_H
#define VTM_RENDER_WORLD_H

#include <vector>

class RenderSystem;
class RenderEntity;
struct D3DXMATRIX;

struct IRenderAction
{
	virtual ~IRenderAction ();

	void PreRender (RenderSystem* r);
	void Render (RenderSystem* r, RenderEntity& e);
	void PostRender (RenderSystem* r);

private:
	virtual void PreRenderImpl (RenderSystem* r) = 0;
	virtual void RenderImpl (RenderSystem* r, RenderEntity& e) = 0;
	virtual void PostRenderImpl (RenderSystem* r) = 0;
};

struct DefaultRenderAction : public IRenderAction
{
private:
	void PreRenderImpl (RenderSystem* r);
	void RenderImpl (RenderSystem* r, RenderEntity& e);
	void PostRenderImpl (RenderSystem* r);
};

class RenderWorld
{
public:
	RenderWorld (RenderSystem* d);

	void AddRenderEntity (std::tr1::shared_ptr<RenderEntity> entity);

	void Render (const D3DXMATRIX& viewProjection);
	void Render (const D3DXMATRIX& viewProjection, IRenderAction* renderAction);

private:
	RenderSystem*	renderSystem_;
	std::vector<std::tr1::shared_ptr<RenderEntity> > entities_;
};

#endif