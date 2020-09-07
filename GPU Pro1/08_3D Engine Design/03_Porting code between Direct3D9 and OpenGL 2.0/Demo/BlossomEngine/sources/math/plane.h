/* $Id: plane.h 126 2009-08-22 17:08:39Z maxest $ */

#ifndef _BLOSSOM_ENGINE_PLANE_
#define _BLOSSOM_ENGINE_PLANE_

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CVector3;
	class CVector4;
	class CMatrix;

	// ----------------------------------------------------------------------------

	class CPlane
	{
	public:
		float a, b, c;
		float d;

	public:
		CPlane();
		CPlane(float a, float b, float c, float d = 0.0f);
		CPlane(const CVector3 &plane);
		CPlane(const CVector4 &plane);
		CPlane(const CVector3 &point, CVector3 normal);
		CPlane(const CVector4 &point, CVector3 normal);
		CPlane(const CVector3 &point1, const CVector3 &point2, const CVector3 &point3); // plane is being normalized
		CPlane(const CVector4 &point1, const CVector4 &point2, const CVector4 &point3); // plane is being normalized

		CVector3 getNormal();
		inline void normalize();

		float getSignedDistanceFromPoint(const CVector3 &point) const; // plane must be normalized
		float getSignedDistanceFromPoint(const CVector4 &point) const; // plane must be normalized

		void transform(const CMatrix &transform); // plane must be normalized; plane is normalized after operation
	};
}

// ----------------------------------------------------------------------------

#endif
