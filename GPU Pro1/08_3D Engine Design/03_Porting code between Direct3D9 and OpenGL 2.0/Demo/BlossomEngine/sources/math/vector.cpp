/* $Id: vector.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#include <cmath>

#include "vector.h"
#include "matrix.h"
#include "math_common.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	CVector2::CVector2()
	{
		x = 0.0f;
		y = 0.0f;
	}



	CVector2::CVector2(float x, float y)
	{
		this->x = x;
		this->y = y;
	}



	CVector2::CVector2(const CVector2 &v)
	{
		x = v.x;
		y = v.y;
	}



	void CVector2::setLength(float length)
	{
		CVector2 temp(x, y);
		temp.normalize();

		x = temp.x * length;
		y = temp.y * length;
	}



	float CVector2::getLength() const
	{
		return sqrtf(x*x + y*y);
	}



	void CVector2::normalize()
	{
		float length = getLength();

		x /= length;
		y /= length;
	}



	CVector2 CVector2::getNormalized() const
	{
		CVector2 temp(x, y);
		temp.normalize();

		return temp;
	}



	float CVector2::operator () (int index) const
	{
		switch (index)
		{

			case 0:
			{
				return x;
			} break;

			case 1:
			{
				return y;
			} break;

			default:
			{
				return 0.0f;
			}

		}
	}



	float CVector2::operator ! () const
	{
		return sqrtf(x*x + y*y);
	}



	float CVector2::operator % (const CVector2 &v) const
	{
		return (x*v.x + y*v.y);
	}



	CVector2 & CVector2::operator = (const CVector2 &v)
	{
		x = v.x;
		y = v.y;

		return *this;
	}



	bool CVector2::operator == (const CVector2 &v) const
	{
		return ( (fabs(x - v.x) < epsilon) && (fabs(y - v.y) < epsilon) );
	}



	bool CVector2::operator != (const CVector2 &v) const
	{
		return !(*this == v);
	}



	CVector2 CVector2::operator + () const
	{
		return CVector2(x, y);
	}



	CVector2 CVector2::operator + (const CVector2 &v) const
	{
		return CVector2(x + v.x, y + v.y);
	}



	CVector2 & CVector2::operator += (const CVector2 &v)
	{
		x += v.x;
		y += v.y;

		return *this;
	}



	CVector2 CVector2::operator - () const
	{
		return CVector2(-x, -y);
	}



	CVector2 CVector2::operator - (const CVector2 &v) const
	{
		return CVector2(x - v.x, y - v.y);
	}



	CVector2 & CVector2::operator -= (const CVector2 &v)
	{
		x -= v.x;
		y -= v.y;

		return *this;
	}



	CVector2 CVector2::operator * (const float &s) const
	{
		return CVector2(x*s, y*s);
	}



	CVector2 & CVector2::operator *= (const float &s)
	{
		x *= s;
		y *= s;

		return *this;
	}



	CVector2 operator * (const float &s, const CVector2 &v)
	{
		return CVector2(s*v.x, s*v.y);
	}



	CVector2 CVector2::operator * (const CVector2 &v) const
	{
		return CVector2(x*v.x, y*v.y);
	}



	CVector2 & CVector2::operator *= (const CVector2 &v)
	{
		x *= v.x;
		y *= v.y;

		return *this;
	}



	// ----------------------------------------------------------------------------



	CVector3::CVector3()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}



	CVector3::CVector3(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}



	CVector3::CVector3(const CVector3 &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}



	CVector3::CVector3(const CVector4 &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}



	void CVector3::setLength(float length)
	{
		CVector3 temp(x, y, z);
		temp.normalize();

		x = temp.x * length;
		y = temp.y * length;
		z = temp.z * length;
	}



	float CVector3::getLength() const
	{
		return sqrtf(x*x + y*y + z*z);
	}



	void CVector3::normalize()
	{
		float length = getLength();

		x /= length;
		y /= length;
		z /= length;
	}



	CVector3 CVector3::getNormalized() const
	{
		CVector3 temp(x, y, z);
		temp.normalize();

		return temp;
	}



	float CVector3::operator () (int index) const
	{
		switch (index)
		{

			case 0:
			{
				return x;
			} break;

			case 1:
			{
				return y;
			} break;

			case 2:
			{
				return z;
			} break;

			default:
			{
				return 0.0f;
			}

		}
	}



	float CVector3::operator ! () const
	{
		return sqrtf(x*x + y*y + z*z);
	}



	float CVector3::operator % (const CVector3 &v) const
	{
		return (x*v.x + y*v.y + z*v.z);
	}



	CVector3 CVector3::operator ^ (const CVector3 &v) const
	{
		return CVector3(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}



	CVector3 & CVector3::operator = (const CVector3 &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;

		return *this;
	}



	bool CVector3::operator == (const CVector3 &v) const
	{
		return ( (fabs(x - v.x) < epsilon) && (fabs(y - v.y) < epsilon) && (fabs(z - v.z) < epsilon) );
	}



	bool CVector3::operator != (const CVector3 &v) const
	{
		return !(*this == v);
	}



	CVector3 & CVector3::operator = (const CVector4 &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;

		return *this;
	}



	bool CVector3::operator == (const CVector4 &v) const
	{
		return ( (fabs(x - v.x) < epsilon) && (fabs(y - v.y) < epsilon) && (fabs(z - v.z) < epsilon) );
	}



	bool CVector3::operator != (const CVector4 &v) const
	{
		return !(*this == v);
	}



	CVector3 CVector3::operator + () const
	{
		return CVector3(x, y, z);
	}



	CVector3 CVector3::operator + (const CVector3 &v) const
	{
		return CVector3(x + v.x, y + v.y, z + v.z);
	}



	CVector3 & CVector3::operator += (const CVector3 &v)
	{
		x += v.x;
		y += v.y;
		z += v.z;

		return *this;
	}



	CVector3 CVector3::operator - () const
	{
		return CVector3(-x, -y, -z);
	}



	CVector3 CVector3::operator - (const CVector3 &v) const
	{
		return CVector3(x - v.x, y - v.y, z - v.z);
	}



	CVector3 & CVector3::operator -= (const CVector3 &v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;

		return *this;
	}



	CVector3 CVector3::operator * (const float &s) const
	{
		return CVector3(x*s, y*s, z*s);
	}



	CVector3 & CVector3::operator *= (const float &s)
	{
		x *= s;
		y *= s;
		z *= s;

		return *this;
	}



	CVector3 operator * (const float &s, const CVector3 &v)
	{
		return CVector3(s*v.x, s*v.y, s*v.z);
	}



	CVector3 CVector3::operator * (const CVector3 &v) const
	{
		return CVector3(x*v.x, y*v.y, z*v.z);
	}



	CVector3 & CVector3::operator *= (const CVector3 &v)
	{
		x *= v.x;
		y *= v.y;
		z *= v.z;

		return *this;
	}



	CVector3 CVector3::operator * (const CMatrix &m) const
	{
		CVector3 temp;

		temp.x = x*m(0, 0) + y*m(1, 0) + z*m(2, 0);
		temp.y = x*m(0, 1) + y*m(1, 1) + z*m(2, 1);
		temp.z = x*m(0, 2) + y*m(1, 2) + z*m(2, 2);

		return temp;
	}



	CVector3 & CVector3::operator *= (const CMatrix &m)
	{
		CVector3 temp;

		temp.x = x*m(0, 0) + y*m(1, 0) + z*m(2, 0);
		temp.y = x*m(0, 1) + y*m(1, 1) + z*m(2, 1);
		temp.z = x*m(0, 2) + y*m(1, 2) + z*m(2, 2);

		*this = temp;
		return *this;
	}



	// ----------------------------------------------------------------------------



	CVector4::CVector4()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
		w = 1.0f;
	}



	CVector4::CVector4(float x, float y, float z, float w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}



	CVector4::CVector4(const CVector4 &v)
	{
		this->x = v.x;
		this->y = v.y;
		this->z = v.z;
		this->w = v.w;
	}



	CVector4::CVector4(const CVector3 &v)
	{
		this->x = v.x;
		this->y = v.y;
		this->z = v.z;
		this->w = 1.0f;
	}



	void CVector4::setLength(float length)
	{
		CVector3 temp(x, y, z);
		temp.normalize();

		x = temp.x * length;
		y = temp.y * length;
		z = temp.z * length;
	}



	float CVector4::getLength() const
	{
		return sqrtf(x*x + y*y + z*z);
	}



	float CVector4::operator () (int index) const
	{
		switch (index)
		{

			case 0:
			{
				return x;
			} break;

			case 1:
			{
				return y;
			} break;

			case 2:
			{
				return z;
			} break;

			case 3:
			{
				return w;
			} break;

			default:
			{
				return 0.0f;
			}

		}
	}



	float CVector4::operator % (const CVector4 &v) const
	{
		return (x*v.x + y*v.y + z*v.z + w*v.w);
	}



	CVector4 & CVector4::operator = (const CVector3 &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
		w = 1.0f;

		return *this;
	}



	bool CVector4::operator == (const CVector3 &v) const
	{
		return ( (fabs(x - v.x) < epsilon) && (fabs(y - v.y) < epsilon) && (fabs(z - v.z) < epsilon) );
	}



	bool CVector4::operator != (const CVector3 &v) const
	{
		return !(*this == v);
	}



	CVector4 & CVector4::operator = (const CVector4 &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
		w = v.w;

		return *this;
	}



	bool CVector4::operator == (const CVector4 &v) const
	{
		return ( (fabs(x - v.x) < epsilon) && (fabs(y - v.y) < epsilon) && (fabs(z - v.z) < epsilon) && (fabs(w - v.w) < epsilon) );
	}



	bool CVector4::operator != (const CVector4 &v) const
	{
		return !(*this == v);
	}



	CVector4 CVector4::operator + (const CVector3 &v) const
	{
		return CVector4(x + v.x, y + v.y, z + v.z, w);
	}



	CVector4 & CVector4::operator += (const CVector3 &v)
	{
		x += v.x;
		y += v.y;
		z += v.z;

		return *this;
	}



	CVector4 CVector4::operator - (const CVector3 &v) const
	{
		return CVector4(x - v.x, y - v.y, z - v.z, w);
	}



	CVector4 & CVector4::operator -= (const CVector3 &v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;

		return *this;
	}



	CVector3 CVector4::operator - (const CVector4 &v) const
	{
		return CVector3(x - v.x, y - v.y, z - v.z);
	}



	CVector4 CVector4::operator * (const CVector4 &v) const
	{
		return CVector4(x*v.x, y*v.y, z*v.z, w*v.w);
	}



	CVector4 & CVector4::operator *= (const CVector4 &v)
	{
		x *= v.x;
		y *= v.y;
		z *= v.z;
		w *= v.w;

		return *this;
	}



	CVector4 CVector4::operator * (const CMatrix &m) const
	{
		CVector4 temp;

		temp.x = x*m(0, 0) + y*m(1, 0) + z*m(2, 0) + m(3, 0);
		temp.y = x*m(0, 1) + y*m(1, 1) + z*m(2, 1) + m(3, 1);
		temp.z = x*m(0, 2) + y*m(1, 2) + z*m(2, 2) + m(3, 2);
		temp.w = x*m(0, 3) + y*m(1, 3) + z*m(2, 3) + m(3, 3);

		float oneOverW = 1.0f/temp.w;
		temp.x *= oneOverW;
		temp.y *= oneOverW;
		temp.z *= oneOverW;
		temp.w = 1.0f;

		return temp;
	}



	CVector4 & CVector4::operator *= (const CMatrix &m)
	{
		CVector4 temp;

		temp.x = x*m(0, 0) + y*m(1, 0) + z*m(2, 0) + m(3, 0);
		temp.y = x*m(0, 1) + y*m(1, 1) + z*m(2, 1) + m(3, 1);
		temp.z = x*m(0, 2) + y*m(1, 2) + z*m(2, 2) + m(3, 2);
		temp.w = x*m(0, 3) + y*m(1, 3) + z*m(2, 3) + m(3, 3);

		float oneOverW = 1.0f/temp.w;
		temp.x *= oneOverW;
		temp.y *= oneOverW;
		temp.z *= oneOverW;
		temp.w = 1.0f;
		
		*this = temp;
		return *this;
	}
}
