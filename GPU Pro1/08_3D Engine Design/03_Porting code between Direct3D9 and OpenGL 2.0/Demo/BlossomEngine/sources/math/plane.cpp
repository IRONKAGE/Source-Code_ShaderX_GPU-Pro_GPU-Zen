/* $Id: plane.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#include <cmath>

#include "plane.h"
#include "vector.h"
#include "matrix.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	CPlane::CPlane()
	{
		a = b = c = d = 0.0f;
	}



	CPlane::CPlane(float a, float b, float c, float d)
	{
		this->a = a;
		this->b = b;
		this->c = c;
		this->d = d;
	}



	CPlane::CPlane(const CVector3 &plane)
	{
		a = plane.x;
		b = plane.y;
		c = plane.z;
		d = 0.0f;
	}



	CPlane::CPlane(const CVector4 &plane)
	{
		a = plane.x;
		b = plane.y;
		c = plane.z;
		d = plane.w;
	}



	CPlane::CPlane(const CVector3 &point, CVector3 normal)
	{
		normal.normalize();

		a = normal.x;
		b = normal.y;
		c = normal.z;
		d = -(point % normal);
	}



	CPlane::CPlane(const CVector4 &point, CVector3 normal)
	{
		normal.normalize();

		a = normal.x;
		b = normal.y;
		c = normal.z;
		d = -((CVector3)point % normal.getNormalized());
	}



	CPlane::CPlane(const CVector3 &point1, const CVector3 &point2, const CVector3 &point3)
	{
		CVector3 v1 = point2 - point1;
		CVector3 v2 = point3 - point1;
		CVector3 n = (v1 ^ v2).getNormalized();

		a = n.x;
		b = n.y;
		c = n.z;
		d = -(point1 % n);
	}



	CPlane::CPlane(const CVector4 &point1, const CVector4 &point2, const CVector4 &point3)
	{
		CVector3 v1 = point2 - point1;
		CVector3 v2 = point3 - point1;
		CVector3 n = (v1 ^ v2).getNormalized();

		a = n.x;
		b = n.y;
		c = n.z;
		d = -(point1 % n);
	}



	CVector3 CPlane::getNormal()
	{
		return CVector3(a, b, c);
	}



	inline void CPlane::normalize()
	{
		float length = sqrtf(a*a + b*b + c*c);

		a /= length;
		b /= length;
		c /= length;
	}



	float CPlane::getSignedDistanceFromPoint(const CVector3 &point) const
	{
		return a*point.x + b*point.y + c*point.z + d;
	}



	float CPlane::getSignedDistanceFromPoint(const CVector4 &point) const
	{
		return a*point.x + b*point.y + c*point.z + d;
	}



	void CPlane::transform(const CMatrix &transform)
	{
		float a, b, c, d;
		transform.getInversed();
		transform.getTransposed();

		a = this->a*transform(0, 0) + this->b*transform(0, 1) + this->c*transform(0, 2) + this->d*transform(0, 3);
		b = this->a*transform(1, 0) + this->b*transform(1, 1) + this->c*transform(1, 2) + this->d*transform(1, 3);
		c = this->a*transform(2, 0) + this->b*transform(2, 1) + this->c*transform(2, 2) + this->d*transform(2, 3);
		d = this->a*transform(3, 0) + this->b*transform(3, 1) + this->c*transform(3, 2) + this->d*transform(3, 3);

		this->a = a;
		this->b = b;
		this->c = c;
		this->d = d;
		
		normalize();
	}
}
