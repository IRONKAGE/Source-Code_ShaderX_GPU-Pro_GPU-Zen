#pragma once

class YawPitchRollCamera
{
	D3DXMATRIX viewMatrix;
	D3DXMATRIX projMatrix;
public:
	YawPitchRollCamera();
	void SetViewParams(D3DXVECTOR3* eyePoint, D3DXVECTOR3* lookAtPoint, D3DXVECTOR3* up);
	void SetProjParams(float fov, float aspect, float front, float back);
	const D3DXMATRIX* GetViewMatrix(){return &viewMatrix;}
	const D3DXMATRIX* GetProjMatrix(){return &projMatrix;}
};
