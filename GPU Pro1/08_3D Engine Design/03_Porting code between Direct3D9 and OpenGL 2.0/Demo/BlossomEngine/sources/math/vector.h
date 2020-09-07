/* $Id: vector.h 126 2009-08-22 17:08:39Z maxest $ */

#ifndef _BLOSSOM_ENGINE_VECTOR_
#define _BLOSSOM_ENGINE_VECTOR_

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CMatrix;
	class CVector2;
	class CVector3;
	class CVector4;

	// ----------------------------------------------------------------------------

	class CVector2
	{
	public:
		float x, y;

	public:
		CVector2();
		CVector2(float x, float y);
		CVector2(const CVector2 &v);

		void setLength(float length);
		float getLength() const;

		void normalize();
		CVector2 getNormalized() const;

		float operator () (int index) const; // retrieves index'th vector's component

		float operator ! () const; // vector's length
		float operator % (const CVector2 &v) const; // dot product

		CVector2 & operator = (const CVector2 &v);
		bool operator == (const CVector2 &v) const;
		bool operator != (const CVector2 &v) const;

		CVector2 operator + () const;
		CVector2 operator + (const CVector2 &v) const;
		CVector2 & operator += (const CVector2 &v);

		CVector2 operator - () const;
		CVector2 operator - (const CVector2 &v) const;
		CVector2 & operator -= (const CVector2 &v);

		CVector2 operator * (const float &s) const;
		CVector2 & operator *= (const float &s);
		friend CVector2 operator * (const float &s, const CVector2 &v);

		CVector2 operator * (const CVector2 &v) const;
		CVector2 & operator *= (const CVector2 &v);
	};

	CVector2 operator * (const float &s, const CVector2 &v);

	// ----------------------------------------------------------------------------

	class CVector3
	{
	public:
		float x, y, z;

	public:
		CVector3();
		CVector3(float x, float y, float z);
		CVector3(const CVector3 &v);
		CVector3(const CVector4 &v);

		void setLength(float length);
		float getLength() const;

		void normalize();
		CVector3 getNormalized() const;

		float operator () (int index) const; // retrieves index'th vector's component

		float operator ! () const; // vector's length
		float operator % (const CVector3 &v) const; // dot product
		CVector3 operator ^ (const CVector3 &v) const; // cross product

		CVector3 & operator = (const CVector3 &v);
		bool operator == (const CVector3 &v) const;
		bool operator != (const CVector3 &v) const;
		CVector3 & operator = (const CVector4 &v);
		bool operator == (const CVector4 &v) const;
		bool operator != (const CVector4 &v) const;

		CVector3 operator + () const;
		CVector3 operator + (const CVector3 &v) const;
		CVector3 & operator += (const CVector3 &v);

		CVector3 operator - () const;
		CVector3 operator - (const CVector3 &v) const;
		CVector3 & operator -= (const CVector3 &v);

		CVector3 operator * (const float &s) const;
		CVector3 & operator *= (const float &s);
		friend CVector3 operator * (const float &s, const CVector3 &v);

		CVector3 operator * (const CVector3 &v) const;
		CVector3 & operator *= (const CVector3 &v);

		CVector3 operator * (const CMatrix &m) const;
		CVector3 & operator *= (const CMatrix &m);
	};

	CVector3 operator * (const float &s, const CVector3 &v);

	// ----------------------------------------------------------------------------

	class CVector4
	{
	public:
		float x, y, z;
		float w;

	public:
		CVector4();
		CVector4(float x, float y, float z, float w = 1.0f);
		CVector4(const CVector4 &v);
		CVector4(const CVector3 &v);

		void setLength(float length);
		float getLength() const;

		float operator () (int index) const; // retrieves index'th vector's component

		float operator % (const CVector4 &v) const; // dot product

		CVector4 & operator = (const CVector3 &v);
		bool operator == (const CVector3 &v) const;
		bool operator != (const CVector3 &v) const;
		CVector4 & operator = (const CVector4 &v);
		bool operator == (const CVector4 &v) const;
		bool operator != (const CVector4 &v) const;

		CVector4 operator + (const CVector3 &v) const;
		CVector4 & operator += (const CVector3 &v);

		CVector4 operator - (const CVector3 &v) const;
		CVector4 & operator -= (const CVector3 &v);

		CVector3 operator - (const CVector4 &v) const;

		CVector4 operator * (const CVector4 &v) const;
		CVector4 & operator *= (const CVector4 &v);

		CVector4 operator * (const CMatrix &m) const;
		CVector4 & operator *= (const CMatrix &m);
	};
}

// ----------------------------------------------------------------------------

#endif
