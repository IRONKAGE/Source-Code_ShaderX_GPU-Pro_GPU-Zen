/* $Id: camera.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#include "../math/blossom_engine_math.h"
#include "camera.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	CCamera::CCamera()
	{
		this->horizontalAngle = 0.0f;
		this->verticalAngle = 0.0f;
		this->distanceFromEyeToAt = 1.0f;
	}



	const CVector3 & CCamera::getEye() const
	{
		return eye;
	}



	const CVector3 & CCamera::getAt() const
	{
		return at;
	}



	const CVector3 & CCamera::getUp() const
	{
		return up;
	}



	const CVector3 & CCamera::getForwardVector() const
	{
		return forwardVector;
	}



	const CVector3 & CCamera::getRightVector() const
	{
		return rightVector;
	}



	const CVector3 & CCamera::getUpVector() const
	{
		return upVector;
	}



	void CCamera::updateFixed(const CVector3 &eye, const CVector3 &at, const CVector3 &up)
	{
		forwardVector = (at - eye).getNormalized();

		this->eye = eye;
		this->at = at;
		this->up = up;
	}



	void CCamera::updateFree(const CVector3 &eye, const CVector3 &up)
	{
		CMatrix transformMatrix = CMatrix::rotateX(verticalAngle) * CMatrix::rotateY(horizontalAngle);

		forwardVector = CVector3(0.0f, 0.0f, -1.0f) * transformMatrix;
		rightVector = CVector3(1.0f, 0.0f, 0.0f) * transformMatrix;
		upVector = rightVector ^ forwardVector;

		this->eye = eye;
		this->at = eye + forwardVector;
		this->up = up;
	}



	void CCamera::updateFocused(const CVector3 &at, const CVector3 &up)
	{
		CMatrix transformMatrix = CMatrix::rotateX(verticalAngle) * CMatrix::rotateY(horizontalAngle);

		forwardVector = CVector3(0.0f, 0.0f, -1.0f) * transformMatrix;
		rightVector = CVector3(1.0f, 0.0f, 0.0f) * transformMatrix;
		upVector = rightVector ^ forwardVector;

		this->eye = at - forwardVector*distanceFromEyeToAt;
		this->at = at;
		this->up = up;
	}
}
