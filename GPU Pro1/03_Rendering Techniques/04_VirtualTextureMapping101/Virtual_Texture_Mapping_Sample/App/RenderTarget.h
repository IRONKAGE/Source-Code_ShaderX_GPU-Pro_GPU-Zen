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
#ifndef VTM_RENDER_TARGET_H
#define VTM_RENDER_TARGET_H

#include <boost/tr1/memory.hpp>

struct ID3D10Texture2D;
struct ID3D10RenderTargetView;
struct ID3D10DepthStencilView;

struct RenderTarget_t
{
	typedef std::tr1::shared_ptr<RenderTarget_t> Ptr;

	~RenderTarget_t ();

	ID3D10Texture2D*				colorRenderTarget;
	ID3D10Texture2D*				depthStencilRenderTarget;

	ID3D10RenderTargetView*			colorRenderTargetView;
	ID3D10DepthStencilView*			depthStencilView;
};

#endif