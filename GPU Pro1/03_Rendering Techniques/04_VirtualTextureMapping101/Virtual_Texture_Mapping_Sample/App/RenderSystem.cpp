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

#include "RenderSystem.h"

#include "RenderTarget.h"
#include "RenderEntity.h"
#include "RenderWorld.h"

/////////////////////////////////////////////////////////////////////////////
void RenderSystem::Init ()
{	
	ID3D10Blob* vtxShader;

	D3DX10CompileFromFile (TEXT("Shaders\\defaultVertex.vts"),
		NULL,
		NULL,
		"VS_Main",
		"vs_4_0",
		D3D10_SHADER_ENABLE_STRICTNESS | D3D10_SHADER_DEBUG,
		0,
		NULL,
		&vtxShader,
		NULL,
		NULL);

	D3D10_INPUT_ELEMENT_DESC inputDesc [] =
	{
		{
			"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
				D3D10_INPUT_PER_VERTEX_DATA, 0
		},
		{
			"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0,
				D3D10_INPUT_PER_VERTEX_DATA, 0
		},
		{
			"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 2, 0,
				D3D10_INPUT_PER_VERTEX_DATA, 0
		}
	};

	device_->CreateInputLayout (inputDesc, 3, vtxShader->GetBufferPointer (), vtxShader->GetBufferSize (),
		&defaultInputLayout_);

	vtxShader->Release ();
}

/////////////////////////////////////////////////////////////////////////////
void RenderSystem::Shutdown ()
{
	defaultInputLayout_->Release ();
}

/////////////////////////////////////////////////////////////////////////////
RenderTarget_t::Ptr RenderSystem::CreateNewRenderTarget(const int width, const int height)
{
	RenderTarget_t::Ptr rt (new RenderTarget_t ());

	D3D10_TEXTURE2D_DESC colorDesc;
	::ZeroMemory (&colorDesc, sizeof(colorDesc));

	//@ TODO Review me
	colorDesc.Width = width;
	colorDesc.Height = height;
	colorDesc.ArraySize = 1;
	colorDesc.SampleDesc.Count = 1;
	colorDesc.SampleDesc.Quality = 0;
	colorDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	colorDesc.BindFlags = D3D10_BIND_RENDER_TARGET;
	colorDesc.Usage = D3D10_USAGE_DEFAULT;
	colorDesc.MipLevels = 1;

	device_->CreateTexture2D (&colorDesc, NULL, &(rt->colorRenderTarget));

	D3D10_TEXTURE2D_DESC depthStencilDesc;
	::ZeroMemory (&depthStencilDesc, sizeof (depthStencilDesc));

	depthStencilDesc.Width = width;
	depthStencilDesc.Height = height;
	depthStencilDesc.ArraySize = 1;
	depthStencilDesc.SampleDesc.Count = 1;
	depthStencilDesc.SampleDesc.Quality = 0;
	depthStencilDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthStencilDesc.BindFlags = D3D10_BIND_DEPTH_STENCIL;
	depthStencilDesc.MipLevels = 1;

	device_->CreateTexture2D (&depthStencilDesc, NULL, &(rt->depthStencilRenderTarget));

	device_->CreateRenderTargetView (rt->colorRenderTarget, NULL, &(rt->colorRenderTargetView));
	device_->CreateDepthStencilView (rt->depthStencilRenderTarget, NULL, &(rt->depthStencilView));

	return rt;
}

/////////////////////////////////////////////////////////////////////////////
void RenderSystem::Render (const RenderEntity& entity, Material& material)
{
	entity.Setup (this);

	// Setup the material
	material.SetWorldViewProjectionMatrix (entity.GetWorldMatrix () * viewProjectionMatrix_);
	material.Bind (device_);
	entity.Render (this);
}

/////////////////////////////////////////////////////////////////////////////
void RenderSystem::Render (const RenderEntity& entity)
{
	Render (entity, *entity.GetMaterial ());
}

/////////////////////////////////////////////////////////////////////////////
RenderSystem::RenderSystem (ID3D10Device* device)
: device_ (device)
{
	defaultRT_ = 0;
	defaultDS_ = 0;
}

/////////////////////////////////////////////////////////////////////////////
RenderSystem::~RenderSystem ()
{
	SAFE_RELEASE(defaultRT_);
	SAFE_RELEASE(defaultDS_);
}

/////////////////////////////////////////////////////////////////////////////
void RenderSystem::OnBeginResize ()
{
	SAFE_RELEASE(defaultRT_);
	SAFE_RELEASE(defaultDS_);
}

/////////////////////////////////////////////////////////////////////////////
void RenderSystem::OnFinishResize ()
{
	device_->OMGetRenderTargets (1, &defaultRT_, &defaultDS_);
	UINT one = 1;
	device_->RSGetViewports (&one, &defaultViewport_);
}

/////////////////////////////////////////////////////////////////////////////
void	RenderSystem::SetRenderTarget (RenderTarget_t& rt)
{
	D3D10_TEXTURE2D_DESC rtDesc;
	rt.colorRenderTarget->GetDesc(&rtDesc);

	D3D10_VIEWPORT vp;
	::ZeroMemory (&vp, sizeof (vp));
	vp.Width = rtDesc.Width;
	vp.Height = rtDesc.Height;
	vp.MaxDepth = 1.0f;

	device_->OMSetRenderTargets (1, &rt.colorRenderTargetView, rt.depthStencilView);
	device_->RSSetViewports (1, &vp);
}

/////////////////////////////////////////////////////////////////////////////
void RenderSystem::Clear (RenderTarget_t& rt, const float color [4], float depth)
{
	device_->ClearRenderTargetView (rt.colorRenderTargetView, color);
	device_->ClearDepthStencilView (rt.depthStencilView, D3D10_CLEAR_DEPTH, depth, 0);
}

/////////////////////////////////////////////////////////////////////////////
void	RenderSystem::SetDefaultRenderTarget ()
{
	device_->RSSetViewports (1, &defaultViewport_);
	device_->OMSetRenderTargets (1, &defaultRT_, defaultDS_);
}

/////////////////////////////////////////////////////////////////////////////
void RenderSystem::SetViewProjectionMatrix (D3DXMATRIX viewProjection)
{
	viewProjectionMatrix_ = viewProjection;
}

/////////////////////////////////////////////////////////////////////////////
ID3D10InputLayout* RenderSystem::GetDefaultLayout ()
{
	return defaultInputLayout_;
}

/////////////////////////////////////////////////////////////////////////////
ID3D10Device* RenderSystem::GetDevice ()
{
	return device_;
}

/////////////////////////////////////////////////////////////////////////////
RenderWorld* RenderSystem::CreateRenderWorld ()
{
	return new RenderWorld (this);
}

/////////////////////////////////////////////////////////////////////////////
void RenderSystem::DestroyRenderWorld (RenderWorld* w)
{
	delete w;
}