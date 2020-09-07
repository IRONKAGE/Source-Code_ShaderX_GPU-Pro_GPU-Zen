#include "dxstdafx.h"
#include "YawPitchRollCamera.h"

YawPitchRollCamera::YawPitchRollCamera(void)
{
	D3DXMatrixIdentity(&viewMatrix);
	D3DXMatrixPerspectiveFovLH(&projMatrix, 1.55f, 1.0f, 0.01f, 10000.0f);
}

void YawPitchRollCamera::SetViewParams(D3DXVECTOR3* eyePoint, D3DXVECTOR3* lookAtPoint, D3DXVECTOR3* up)
{
	D3DXMatrixLookAtLH(&viewMatrix, eyePoint, lookAtPoint, up);
}

void YawPitchRollCamera::SetProjParams(float fov, float aspect, float front, float back)
{
	D3DXMatrixPerspectiveFovLH(&projMatrix, fov, aspect, front, back);
}