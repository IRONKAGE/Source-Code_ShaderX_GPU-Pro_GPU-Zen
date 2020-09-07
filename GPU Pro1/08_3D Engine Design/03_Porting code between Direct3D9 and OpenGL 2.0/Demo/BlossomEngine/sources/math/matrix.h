/* $Id: matrix.h 126 2009-08-22 17:08:39Z maxest $ */

#ifndef _BLOSSOM_ENGINE_MATRIX_
#define _BLOSSOM_ENGINE_MATRIX_

// ----------------------------------------------------------------------------

namespace Blossom
{
	class CVector3;
	class CVector4;

	// ----------------------------------------------------------------------------

	class CMatrix
	{
	public:
		float elements[4][4];

	public:
		static CMatrix identity();

		static CMatrix lookAtLH(const CVector3 &eye, const CVector3 &at, const CVector3 &up);
		static CMatrix lookAtRH(const CVector3 &eye, const CVector3 &at, const CVector3 &up);
		static CMatrix perspectiveFovLH(float fovY, float aspectRatio, float zNear, float zFar); // zFar == 0 -> far plane at infinity
		static CMatrix perspectiveFovRH(float fovY, float aspectRatio, float zNear, float zFar); // zFar == 0 -> far plane at infinity
		static CMatrix orthoOffCenterLH(float l, float r, float b, float t, float zNear, float zFar);
		static CMatrix orthoOffCenterRH(float l, float r, float b, float t, float zNear, float zFar);
		static CMatrix orthoLH(float width, float height, float zNear, float zFar);
		static CMatrix orthoRH(float width, float height, float zNear, float zFar);

		static CMatrix translate(float tx, float ty, float tz);
		static CMatrix rotate(float angle, float rx, float ry, float rz);
		static CMatrix scale(float sx, float sy, float sz);

		static CMatrix reflect(const CVector3 &planePoint, const CVector3 &planeNormal);

		static CMatrix rotateX(float angle);
		static CMatrix rotateY(float angle);
		static CMatrix rotateZ(float angle);

	public:
		CMatrix();
		CMatrix(float _00, float _01, float _02, float _03,
				float _10, float _11, float _12, float _13,
				float _20, float _21, float _22, float _23,
				float _30, float _31, float _32, float _33);
		CMatrix(const CMatrix &m);

		void setMatrix(float _00, float _01, float _02, float _03,
					   float _10, float _11, float _12, float _13,
					   float _20, float _21, float _22, float _23,
					   float _30, float _31, float _32, float _33);
		void setMatrix(const CMatrix &m);
		CMatrix getMatrix() const;

		bool isOrthonormal();

		void transpose();
		CMatrix getTransposed() const;

		void inverse();
		CMatrix getInversed() const;

		void loadIdentity();

		void loadLookAtLH(const CVector3 &eye, const CVector3 &at, const CVector3 &up);
		void loadLookAtRH(const CVector3 &eye, const CVector3 &at, const CVector3 &up);
		void loadPerspectiveFovLH(float fovY, float aspectRatio, float zNear, float zFar);
		void loadPerspectiveFovRH(float fovY, float aspectRatio, float zNear, float zFar);
		void loadOrthoOffCenterLH(float l, float r, float b, float t, float zNear, float zFar);
		void loadOrthoOffCenterRH(float l, float r, float b, float t, float zNear, float zFar);
		void loadOrthoLH(float width, float height, float zNear, float zFar);
		void loadOrthoRH(float width, float height, float zNear, float zFar);

		void loadTranslate(float tx, float ty, float tz);
		void loadRotate(float angle, float rx, float ry, float rz);
		void loadScale(float sx, float sy, float sz);

		void loadReflect(const CVector3 &planePoint, const CVector3 &planeNormal);

		void loadRotateX(float angle);
		void loadRotateY(float angle);
		void loadRotateZ(float angle);

		float operator () (int row, int col) const;
		float & operator () (int row, int col);

		CMatrix & operator = (const CMatrix &m);
		bool operator == (const CMatrix &m) const;
		bool operator != (const CMatrix &m) const;

		CMatrix operator * (const CMatrix &m) const;
		CMatrix & operator *= (const CMatrix &m);

		CVector3 operator * (const CVector3 &v) const;
		CVector4 operator * (const CVector4 &v) const;
	};
}

// ----------------------------------------------------------------------------

#endif
