/* $Id: camera.h 216 2009-09-02 14:03:48Z maxest $ */

#ifndef _BLOSSOM_ENGINE_CAMERA_
#define _BLOSSOM_ENGINE_CAMERA_

#include "../math/blossom_engine_math.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CCamera
	{
	private:
		CVector3 eye, at, up;
		CVector3 forwardVector, rightVector, upVector;

	public:
		float horizontalAngle, verticalAngle;
		float distanceFromEyeToAt;

	public:
		CCamera();

		const CVector3 & getEye() const;
		const CVector3 & getAt() const;
		const CVector3 & getUp() const;

		const CVector3 & getForwardVector() const;
		const CVector3 & getRightVector() const;
		const CVector3 & getUpVector() const;

		void updateFixed(const CVector3 &eye, const CVector3 &at, const CVector3 &up = CVector3(0.0f, 1.0f, 0.0f));
		void updateFree(const CVector3 &eye, const CVector3 &up = CVector3(0.0f, 1.0f, 0.0f));
		void updateFocused(const CVector3 &at, const CVector3 &up = CVector3(0.0f, 1.0f, 0.0f));
	};
}

// ----------------------------------------------------------------------------

#endif
