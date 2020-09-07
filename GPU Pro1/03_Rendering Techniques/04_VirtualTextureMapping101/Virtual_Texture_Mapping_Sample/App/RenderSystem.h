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
#ifndef VTM_RENDER_HELPER_H
#define VTM_RENDER_HELPER_H

#include <D3D10.h>
#include <D3DX10Math.h>
#include <boost/tr1/memory.hpp>

#include "Material.h"

class RenderEntity;
struct RenderTarget_t;
class RenderWorld;

class RenderSystem
{
public:
	RenderSystem (ID3D10Device* device);

	~RenderSystem ();

	RenderWorld*	CreateRenderWorld ();
	void			DestroyRenderWorld (RenderWorld* world);

	/**
	* Call this each time the window is about to change
	*/
	void OnBeginResize ();

	/**
	* Call this after the window has been changed
	*/
	void OnFinishResize ();

	std::tr1::shared_ptr<RenderTarget_t>
		CreateNewRenderTarget (const int width, const int height);

	void	SetRenderTarget (RenderTarget_t& rt);

	void Clear (RenderTarget_t& rt, const float color [4], float depth = 1.0f);

	/**
	* Set the render target which was active during the RenderSystem creation.
	*/
	void	SetDefaultRenderTarget ();

	void Init ();

	void Shutdown ();

	void SetViewProjectionMatrix (D3DXMATRIX viewProjection);

	void Render (const RenderEntity& entity, Material& material);
	void Render (const RenderEntity& entity);

	ID3D10InputLayout* GetDefaultLayout ();

	ID3D10Device* GetDevice ();

private:
	ID3D10Device*	device_;
	D3DMATRIX		viewProjectionMatrix_;
	D3D10_VIEWPORT	defaultViewport_;

	ID3D10InputLayout* defaultInputLayout_;

	ID3D10RenderTargetView* defaultRT_;
	ID3D10DepthStencilView* defaultDS_;
};

#endif