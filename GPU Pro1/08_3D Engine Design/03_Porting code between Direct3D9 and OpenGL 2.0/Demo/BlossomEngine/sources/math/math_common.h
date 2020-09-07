/* $Id: math_common.h 167 2009-08-25 14:03:17Z maxest $ */

#ifndef _BLOSSOM_ENGINE_MATH_COMMON_
#define _BLOSSOM_ENGINE_MATH_COMMON_

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CVector2;
	class CVector3;

	// ----------------------------------------------------------------------------

	const float PI = 3.141593f;
	const float epsilon = 0.0001f;

	// ----------------------------------------------------------------------------

	#define ROUND(x) (float)(int)(x + 0.5f)

	// ----------------------------------------------------------------------------

	float saturate(float value);

	float deg2rad(float degrees);
	float rad2deg(float radians);

	float getDistanceBetweenPoints(const CVector2 &v1, const CVector2 &v2);
	float getAngleBetweenVectors(const CVector2 &v1, const CVector2 &v2);
	CVector2 getReflectedVector(CVector2 input, CVector2 normal);

	float getDistanceBetweenPoints(const CVector3 &v1, const CVector3 &v2);
	float getAngleBetweenVectors(const CVector3 &v1, const CVector3 &v2);
	CVector3 getReflectedVector(CVector3 input, CVector3 normal);

	void computeTangentBasisForTriangle(
		const CVector3 &v1, const CVector2 &uv1,
		const CVector3 &v2, const CVector2 &uv2,
		const CVector3 &v3, const CVector2 &uv3,
		CVector3 &tangent, CVector3 &bitangent, CVector3 &normal);
}

// ----------------------------------------------------------------------------

#endif
