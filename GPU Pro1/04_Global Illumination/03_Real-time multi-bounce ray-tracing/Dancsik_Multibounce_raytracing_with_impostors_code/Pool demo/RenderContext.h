#pragma once
#include "UnicodeString.h"

class RenderContext
{
public:

	LPDIRECT3DDEVICE9 device;
	LPD3DXEFFECT effect;
	const D3DXMATRIX* projMatrix;
	const D3DXMATRIX* viewMatrix;
	const D3DXMATRIX* nodeTransformMatrix;
	const UnicodeString roleName;
	RenderContext(
		LPDIRECT3DDEVICE9 device,
		LPD3DXEFFECT effect,
		const D3DXMATRIX* projMatrix,
		const D3DXMATRIX* viewMatrix,
		const D3DXMATRIX* nodeTransformMatrix,
		const UnicodeString& roleName)
		:roleName(roleName)
	{
		this->device = device;
		this->effect = effect;
		this->projMatrix = projMatrix;
		this->viewMatrix = viewMatrix;
		this->nodeTransformMatrix = nodeTransformMatrix;
	}
};