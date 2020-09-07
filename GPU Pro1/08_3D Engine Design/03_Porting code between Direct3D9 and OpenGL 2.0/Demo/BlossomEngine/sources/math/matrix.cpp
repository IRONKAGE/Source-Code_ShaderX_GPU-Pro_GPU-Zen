/* $Id: matrix.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#include <cmath>

#include "matrix.h"
#include "vector.h"
#include "math_common.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	CMatrix CMatrix::identity()
	{
		CMatrix temp;
		temp.loadIdentity();

		return temp;
	}



	CMatrix CMatrix::lookAtLH(const CVector3 &eye, const CVector3 &at, const CVector3 &up)
	{
		CMatrix temp;
		temp.loadLookAtLH(eye, at, up);

		return temp;
	}



	CMatrix CMatrix::lookAtRH(const CVector3 &eye, const CVector3 &at, const CVector3 &up)
	{
		CMatrix temp;
		temp.loadLookAtRH(eye, at, up);

		return temp;
	}



	CMatrix CMatrix::perspectiveFovLH(float fovY, float aspectRatio, float zNear, float zFar)
	{
		CMatrix temp;
		temp.loadPerspectiveFovLH(fovY, aspectRatio, zNear, zFar);

		return temp;
	}



	CMatrix CMatrix::perspectiveFovRH(float fovY, float aspectRatio, float zNear, float zFar)
	{
		CMatrix temp;
		temp.loadPerspectiveFovRH(fovY, aspectRatio, zNear, zFar);

		return temp;
	}



	CMatrix CMatrix::orthoOffCenterLH(float l, float r, float b, float t, float zNear, float zFar)
	{
		CMatrix temp;
		temp.loadOrthoOffCenterLH(l, r, b, t, zNear, zFar);

		return temp;
	}



	CMatrix CMatrix::orthoOffCenterRH(float l, float r, float b, float t, float zNear, float zFar)
	{
		CMatrix temp;
		temp.loadOrthoOffCenterRH(l, r, b, t, zNear, zFar);

		return temp;
	}



	CMatrix CMatrix::orthoLH(float width, float height, float zNear, float zFar)
	{
		CMatrix temp;
		temp.loadOrthoLH(width, height, zNear, zFar);

		return temp;
	}



	CMatrix CMatrix::orthoRH(float width, float height, float zNear, float zFar)
	{
		CMatrix temp;
		temp.loadOrthoRH(width, height, zNear, zFar);

		return temp;
	}



	CMatrix CMatrix::translate(float tx, float ty, float tz)
	{
		CMatrix temp;
		temp.loadTranslate(tx, ty, tz);

		return temp;
	}



	CMatrix CMatrix::rotate(float angle, float rx, float ry, float rz)
	{
		CMatrix temp;
		temp.loadRotate(angle, rx, ry, rz);

		return temp;
	}



	CMatrix CMatrix::scale(float sx, float sy, float sz)
	{
		CMatrix temp;
		temp.loadScale(sx, sy, sz);

		return temp;
	}



	CMatrix CMatrix::reflect(const CVector3 &planePoint, const CVector3 &planeNormal)
	{
		CMatrix temp;
		temp.loadReflect(planePoint, planeNormal);

		return temp;
	}



	CMatrix CMatrix::rotateX(float angle)
	{
		CMatrix temp;
		temp.loadRotateX(angle);

		return temp;
	}



	CMatrix CMatrix::rotateY(float angle)
	{
		CMatrix temp;
		temp.loadRotateY(angle);

		return temp;
	}



	CMatrix CMatrix::rotateZ(float angle)
	{
		CMatrix temp;
		temp.loadRotateZ(angle);

		return temp;
	}



	CMatrix::CMatrix()
	{
		elements[0][0] = 1.0f;	elements[0][1] = 0.0f;	elements[0][2] = 0.0f;	elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;	elements[1][1] = 1.0f;	elements[1][2] = 0.0f;	elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;	elements[2][1] = 0.0f;	elements[2][2] = 1.0f;	elements[2][3] = 0.0f;
		elements[3][0] = 0.0f;	elements[3][1] = 0.0f;	elements[3][2] = 0.0f;	elements[3][3] = 1.0f;
	}



	CMatrix::CMatrix(float _00, float _01, float _02, float _03,
					 float _10, float _11, float _12, float _13,
					 float _20, float _21, float _22, float _23,
					 float _30, float _31, float _32, float _33)
	{
		elements[0][0] = _00;	elements[0][1] = _01;	elements[0][2] = _02;	elements[0][3] = _03;
		elements[1][0] = _10;	elements[1][1] = _11;	elements[1][2] = _12;	elements[1][3] = _13;
		elements[2][0] = _20;	elements[2][1] = _21;	elements[2][2] = _22;	elements[2][3] = _23;
		elements[3][0] = _30;	elements[3][1] = _31;	elements[3][2] = _32;	elements[3][3] = _33;
	}



	CMatrix::CMatrix(const CMatrix &m)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				elements[i][j] = m.elements[i][j];
			}
		}
	}



	void CMatrix::setMatrix(float _00, float _01, float _02, float _03,
							float _10, float _11, float _12, float _13,
							float _20, float _21, float _22, float _23,
							float _30, float _31, float _32, float _33)
	{
		elements[0][0] = _00;	elements[0][1] = _01;	elements[0][2] = _02;	elements[0][3] = _03;
		elements[1][0] = _10;	elements[1][1] = _11;	elements[1][2] = _12;	elements[1][3] = _13;
		elements[2][0] = _20;	elements[2][1] = _21;	elements[2][2] = _22;	elements[2][3] = _23;
		elements[3][0] = _30;	elements[3][1] = _31;	elements[3][2] = _32;	elements[3][3] = _33;
	}



	void CMatrix::setMatrix(const CMatrix &m)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				elements[i][j] = m.elements[i][j];
			}
		}
	}



	CMatrix CMatrix::getMatrix() const
	{
		return CMatrix(*this);
	}



	bool CMatrix::isOrthonormal()
	{
		CVector4 row1(elements[0][0], elements[0][1], elements[0][2], elements[0][3]);
		CVector4 row2(elements[1][0], elements[1][1], elements[1][2], elements[1][3]);
		CVector4 row3(elements[2][0], elements[2][1], elements[2][2], elements[2][3]);
		CVector4 row4(elements[3][0], elements[3][1], elements[3][2], elements[3][3]);

		if ( (row1 % row2 < epsilon) &&
			 (row1 % row3 < epsilon) &&
			 (row1 % row4 < epsilon) &&
			 (row2 % row3 < epsilon) &&
			 (row2 % row4 < epsilon) &&
			 (row3 % row4 < epsilon) )
		{
			return true;
		}
		else
		{
			return false;
		}
	}



	void CMatrix::transpose()
	{
		CMatrix temp = getMatrix();

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				elements[i][j] = temp.elements[j][i];
			}
		}
	}



	CMatrix CMatrix::getTransposed() const
	{
		CMatrix temp = getMatrix();
		temp.transpose();

		return temp;
	}



	// some helper function for the one below
	float inline getDeterminant(float a, float b, float c,
								float d, float e, float f,
								float g, float h, float i)
	{
		return ( (a*e*i + d*h*c + g*b*f) - (c*e*g + f*h*a + i*b*d) );
	}

	// yeah, I know - that's horrible function! That's all Laplace fault!
	void CMatrix::inverse()
	{
		float determinant =
							+ elements[0][0]*getDeterminant(elements[1][1], elements[1][2], elements[1][3],
															elements[2][1], elements[2][2], elements[2][3],
															elements[3][1], elements[3][2], elements[3][3])
							- elements[0][1]*getDeterminant(elements[1][0], elements[1][2], elements[1][3],
															elements[2][0], elements[2][2], elements[2][3],
															elements[3][0], elements[3][2], elements[3][3])
							+ elements[0][2]*getDeterminant(elements[1][0], elements[1][1], elements[1][3],
															elements[2][0], elements[2][1], elements[2][3],
															elements[3][0], elements[3][1], elements[3][3])
							- elements[0][3]*getDeterminant(elements[1][0], elements[1][1], elements[1][2],
															elements[2][0], elements[2][1], elements[2][2],
															elements[3][0], elements[3][1], elements[3][2]);

		if ( !(fabs(determinant) < epsilon) )
		{
			float adjElements[4][4];

			adjElements[0][0] = + getDeterminant(elements[1][1], elements[1][2], elements[1][3],
												 elements[2][1], elements[2][2], elements[2][3],
												 elements[3][1], elements[3][2], elements[3][3]);
			adjElements[0][1] = - getDeterminant(elements[1][0], elements[1][2], elements[1][3],
												 elements[2][0], elements[2][2], elements[2][3],
												 elements[3][0], elements[3][2], elements[3][3]);
			adjElements[0][2] = + getDeterminant(elements[1][0], elements[1][1], elements[1][3],
												 elements[2][0], elements[2][1], elements[2][3],
												 elements[3][0], elements[3][1], elements[3][3]);
			adjElements[0][3] = - getDeterminant(elements[1][0], elements[1][1], elements[1][2],
												 elements[2][0], elements[2][1], elements[2][2],
												 elements[3][0], elements[3][1], elements[3][2]);

			adjElements[1][0] = - getDeterminant(elements[0][1], elements[0][2], elements[0][3],
												 elements[2][1], elements[2][2], elements[2][3],
												 elements[3][1], elements[3][2], elements[3][3]);
			adjElements[1][1] = + getDeterminant(elements[0][0], elements[0][2], elements[0][3],
												 elements[2][0], elements[2][2], elements[2][3],
												 elements[3][0], elements[3][2], elements[3][3]);
			adjElements[1][2] = - getDeterminant(elements[0][0], elements[0][1], elements[0][3],
												 elements[2][0], elements[2][1], elements[2][3],
												 elements[3][0], elements[3][1], elements[3][3]);
			adjElements[1][3] = + getDeterminant(elements[0][0], elements[0][1], elements[0][2],
												 elements[2][0], elements[2][1], elements[2][2],
												 elements[3][0], elements[3][1], elements[3][2]);

			adjElements[2][0] = + getDeterminant(elements[0][1], elements[0][2], elements[0][3],
												 elements[1][1], elements[1][2], elements[1][3],
												 elements[3][1], elements[3][2], elements[3][3]);
			adjElements[2][1] = - getDeterminant(elements[0][0], elements[0][2], elements[0][3],
												 elements[1][0], elements[1][2], elements[1][3],
												 elements[3][0], elements[3][2], elements[3][3]);
			adjElements[2][2] = + getDeterminant(elements[0][0], elements[0][1], elements[0][3],
												 elements[1][0], elements[1][1], elements[1][3],
												 elements[3][0], elements[3][1], elements[3][3]);
			adjElements[2][3] = - getDeterminant(elements[0][0], elements[0][1], elements[0][2],
												 elements[1][0], elements[1][1], elements[1][2],
												 elements[3][0], elements[3][1], elements[3][2]);

			adjElements[3][0] = - getDeterminant(elements[0][1], elements[0][2], elements[0][3],
												 elements[1][1], elements[1][2], elements[1][3],
												 elements[2][1], elements[2][2], elements[2][3]);
			adjElements[3][1] = + getDeterminant(elements[0][0], elements[0][2], elements[0][3],
												 elements[1][0], elements[1][2], elements[1][3],
												 elements[2][0], elements[2][2], elements[2][3]);
			adjElements[3][2] = - getDeterminant(elements[0][0], elements[0][1], elements[0][3],
												 elements[1][0], elements[1][1], elements[1][3],
												 elements[2][0], elements[2][1], elements[2][3]);
			adjElements[3][3] = + getDeterminant(elements[0][0], elements[0][1], elements[0][2],
												 elements[1][0], elements[1][1], elements[1][2],
												 elements[2][0], elements[2][1], elements[2][2]);

			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					elements[i][j] = (1.0f/determinant) * adjElements[j][i];
				}
			}
		}
	}



	CMatrix CMatrix::getInversed() const
	{
		CMatrix temp = getMatrix();
		temp.inverse();

		return temp;
	}



	void CMatrix::loadIdentity()
	{
		elements[0][0] = 1.0f;	elements[0][1] = 0.0f;	elements[0][2] = 0.0f;	elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;	elements[1][1] = 1.0f;	elements[1][2] = 0.0f;	elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;	elements[2][1] = 0.0f;	elements[2][2] = 1.0f;	elements[2][3] = 0.0f;
		elements[3][0] = 0.0f;	elements[3][1] = 0.0f;	elements[3][2] = 0.0f;	elements[3][3] = 1.0f;
	}



	void CMatrix::loadLookAtLH(const CVector3 &eye, const CVector3 &at, const CVector3 &up)
	{
		CVector3 zAxis = (at - eye).getNormalized();
		CVector3 xAxis = (up ^ zAxis).getNormalized();
		CVector3 yAxis = (zAxis ^ xAxis);

		elements[0][0] = xAxis.x;			elements[0][1] = yAxis.x;			elements[0][2] = zAxis.x;			elements[0][3] = 0.0f;
		elements[1][0] = xAxis.y;			elements[1][1] = yAxis.y;			elements[1][2] = zAxis.y;			elements[1][3] = 0.0f;
		elements[2][0] = xAxis.z;			elements[2][1] = yAxis.z;			elements[2][2] = zAxis.z;			elements[2][3] = 0.0f;
		elements[3][0] = -(xAxis % eye);	elements[3][1] = -(yAxis % eye);	elements[3][2] = -(zAxis % eye);	elements[3][3] = 1.0f;
	}



	void CMatrix::loadLookAtRH(const CVector3 &eye, const CVector3 &at, const CVector3 &up)
	{
		CVector3 zAxis = (eye - at).getNormalized();
		CVector3 xAxis = (up ^ zAxis).getNormalized();
		CVector3 yAxis = (zAxis ^ xAxis);

		elements[0][0] = xAxis.x;			elements[0][1] = yAxis.x;			elements[0][2] = zAxis.x;			elements[0][3] = 0.0f;
		elements[1][0] = xAxis.y;			elements[1][1] = yAxis.y;			elements[1][2] = zAxis.y;			elements[1][3] = 0.0f;
		elements[2][0] = xAxis.z;			elements[2][1] = yAxis.z;			elements[2][2] = zAxis.z;			elements[2][3] = 0.0f;
		elements[3][0] = -(xAxis % eye);	elements[3][1] = -(yAxis % eye);	elements[3][2] = -(zAxis % eye);	elements[3][3] = 1.0f;
	}



	void CMatrix::loadPerspectiveFovLH(float fovY, float aspectRatio, float zNear, float zFar)
	{
		float yScale = 1.0f / tanf(fovY / 2.0f);
		float xScale = yScale / aspectRatio;

		elements[0][0] = xScale;	elements[0][1] = 0.0f;		elements[0][2] = 0.0f;								elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;		elements[1][1] = yScale;	elements[1][2] = 0.0f;								elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;		elements[2][1] = 0.0f;		elements[2][2] = zFar / (zFar - zNear);				elements[2][3] = 1.0f;
		elements[3][0] = 0.0f;		elements[3][1] = 0.0f;		elements[3][2] = -zNear * zFar / (zFar - zNear);	elements[3][3] = 0.0f;

		if (zFar == 0.0f)
		{
			elements[2][2] = 1.0f;			
			elements[3][2] = -zNear;
		}
	}



	void CMatrix::loadPerspectiveFovRH(float fovY, float aspectRatio, float zNear, float zFar)
	{
		float yScale = 1.0f / tanf(fovY / 2.0f);
		float xScale = yScale / aspectRatio;

		elements[0][0] = xScale;	elements[0][1] = 0.0f;		elements[0][2] = 0.0f;								elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;		elements[1][1] = yScale;	elements[1][2] = 0.0f;								elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;		elements[2][1] = 0.0f;		elements[2][2] = zFar / (zNear - zFar);				elements[2][3] = -1.0f;
		elements[3][0] = 0.0f;		elements[3][1] = 0.0f;		elements[3][2] = zNear * zFar / (zNear - zFar);		elements[3][3] = 0.0f;

		if (zFar == 0.0f)
		{
			elements[2][2] = -1.0f;			
			elements[3][2] = -zNear;
		}
	}



	void CMatrix::loadOrthoOffCenterLH(float l, float r, float b, float t, float zNear, float zFar)
	{
		elements[0][0] = 2.0f / (r - l);		elements[0][1] = 0.0f;				elements[0][2] = 0.0f;						elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;					elements[1][1] = 2.0f / (t - b);	elements[1][2] = 0.0f;						elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;					elements[2][1] = 0.0f;				elements[2][2] = 1.0f / (zFar - zNear);		elements[2][3] = 0.0f;
		elements[3][0] = (1.0f + r)/(1.0f - r);	elements[3][1] = (t + b)/(b - t);	elements[3][2] = -zNear / (zFar - zNear);	elements[3][3] = 1.0f;
	}



	void CMatrix::loadOrthoOffCenterRH(float l, float r, float b, float t, float zNear, float zFar)
	{
		elements[0][0] = 2.0f / (r - l);		elements[0][1] = 0.0f;				elements[0][2] = 0.0f;						elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;					elements[1][1] = 2.0f / (t - b);	elements[1][2] = 0.0f;						elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;					elements[2][1] = 0.0f;				elements[2][2] = 1.0f / (zNear - zFar);		elements[2][3] = 0.0f;
		elements[3][0] = (1.0f + r)/(1.0f - r);	elements[3][1] = (t + b)/(b - t);	elements[3][2] = -zNear / (zFar - zNear);	elements[3][3] = 1.0f;
	}



	void CMatrix::loadOrthoLH(float width, float height, float zNear, float zFar)
	{
		elements[0][0] = 2.0f / width;	elements[0][1] = 0.0f;				elements[0][2] = 0.0f;						elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;			elements[1][1] = 2.0f / height;		elements[1][2] = 0.0f;						elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;			elements[2][1] = 0.0f;				elements[2][2] = 1.0f / (zFar - zNear);		elements[2][3] = 0.0f;
		elements[3][0] = 0.0f;			elements[3][1] = 0.0f;				elements[3][2] = -zNear / (zFar - zNear);	elements[3][3] = 1.0f;
	}



	void CMatrix::loadOrthoRH(float width, float height, float zNear, float zFar)
	{
		elements[0][0] = 2.0f / width;	elements[0][1] = 0.0f;				elements[0][2] = 0.0f;						elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;			elements[1][1] = 2.0f / height;		elements[1][2] = 0.0f;						elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;			elements[2][1] = 0.0f;				elements[2][2] = 1.0f / (zNear - zFar);		elements[2][3] = 0.0f;
		elements[3][0] = 0.0f;			elements[3][1] = 0.0f;				elements[3][2] = zNear / (zNear - zFar);	elements[3][3] = 1.0f;
	}



	void CMatrix::loadTranslate(float tx, float ty, float tz)
	{
		elements[0][0] = 1.0f;	elements[0][1] = 0.0f;	elements[0][2] = 0.0f;	elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;	elements[1][1] = 1.0f;	elements[1][2] = 0.0f;	elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;	elements[2][1] = 0.0f;	elements[2][2] = 1.0f;	elements[2][3] = 0.0f;
		elements[3][0] = tx;	elements[3][1] = ty;	elements[3][2] = tz;	elements[3][3] = 1.0f;
	}



	void CMatrix::loadRotate(float angle, float rx, float ry, float rz)
	{
		float s = sinf(angle);
		float c = cosf(angle);

		float length = sqrtf(rx*rx + ry*ry + rz*rz);

		rx /= length;
		ry /= length;
		rz /= length;

		elements[0][0] = c+rx*rx*(1-c);			elements[0][1] = rx*ry*(1-c)+rz*s;		elements[0][2] = rx*rz*(1-c)-ry*s;		elements[0][3] = 0.0f;
		elements[1][0] = rx*ry*(1-c)-rz*s;		elements[1][1] = c+ry*ry*(1-c);			elements[1][2] = ry*rz*(1-c)+rx*s;		elements[1][3] = 0.0f;
		elements[2][0] = rx*rz*(1-c)+ry*s;		elements[2][1] = ry*rz*(1-c)-rx*s;		elements[2][2] = c+rz*rz*(1-c);			elements[2][3] = 0.0f;
		elements[3][0] = 0.0f;					elements[3][1] = 0.0f;					elements[3][2] = 0.0f;					elements[3][3] = 1.0f;
	}



	void CMatrix::loadScale(float sx, float sy, float sz)
	{
		elements[0][0] = sx;	elements[0][1] = 0.0f;	elements[0][2] = 0.0f;	elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;	elements[1][1] = sy;	elements[1][2] = 0.0f;	elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;	elements[2][1] = 0.0f;	elements[2][2] = sz;	elements[2][3] = 0.0f;
		elements[3][0] = 0.0f;	elements[3][1] = 0.0f;	elements[3][2] = 0.0f;	elements[3][3] = 1.0f;
	}



	void CMatrix::loadReflect(const CVector3 &planePoint, const CVector3 &planeNormal)
	{
		CVector3 normalizedPlaneNormal = planeNormal;
		normalizedPlaneNormal.normalize();

		float planeD = -(planePoint % normalizedPlaneNormal);

		elements[0][0] = -2.0f*normalizedPlaneNormal.x*normalizedPlaneNormal.x + 1.0f;	elements[0][1] = -2.0f*normalizedPlaneNormal.y*normalizedPlaneNormal.x;			elements[0][2] = -2.0f*normalizedPlaneNormal.z*normalizedPlaneNormal.x;			elements[0][3] = 0.0f;
		elements[1][0] = -2.0f*normalizedPlaneNormal.x*normalizedPlaneNormal.y;			elements[1][1] = -2.0f*normalizedPlaneNormal.y*normalizedPlaneNormal.y + 1.0f;	elements[1][2] = -2.0f*normalizedPlaneNormal.z*normalizedPlaneNormal.y;			elements[1][3] = 0.0f;
		elements[2][0] = -2.0f*normalizedPlaneNormal.x*normalizedPlaneNormal.z;			elements[2][1] = -2.0f*normalizedPlaneNormal.y*normalizedPlaneNormal.z;			elements[2][2] = -2.0f*normalizedPlaneNormal.z*normalizedPlaneNormal.z + 1.0f;	elements[2][3] = 0.0f;
		elements[3][0] = -2.0f*normalizedPlaneNormal.x*planeD + 1.0f;					elements[3][1] = -2.0f*normalizedPlaneNormal.y*planeD;							elements[3][2] = -2.0f*normalizedPlaneNormal.z*planeD;							elements[3][3] = 1.0f;
	}



	void CMatrix::loadRotateX(float angle)
	{
		elements[0][0] = 1.0f;	elements[0][1] = 0.0f;			elements[0][2] = 0.0f;			elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;	elements[1][1] = cosf(angle);	elements[1][2] = sin(angle);	elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;	elements[2][1] = -sin(angle);	elements[2][2] = cos(angle);	elements[2][3] = 0.0f;
		elements[3][0] = 0.0f;	elements[3][1] = 0.0f;			elements[3][2] = 0.0f;			elements[3][3] = 1.0f;
	}



	void CMatrix::loadRotateY(float angle)
	{
		elements[0][0] = cos(angle);	elements[0][1] = 0.0f;	elements[0][2] = -sin(angle);	elements[0][3] = 0.0f;
		elements[1][0] = 0.0f;			elements[1][1] = 1.0f;	elements[1][2] = 0.0f;			elements[1][3] = 0.0f;
		elements[2][0] = sin(angle);	elements[2][1] = 0.0f;	elements[2][2] = cos(angle);	elements[2][3] = 0.0f;
		elements[3][0] = 0.0f;			elements[3][1] = 0.0f;	elements[3][2] = 0.0f;			elements[3][3] = 1.0f;
	}



	void CMatrix::loadRotateZ(float angle)
	{
		elements[0][0] = cos(angle);	elements[0][1] = sin(angle);	elements[0][2] = 0.0f;	elements[0][3] = 0.0f;
		elements[1][0] = -sin(angle);	elements[1][1] = cos(angle);	elements[1][2] = 0.0f;	elements[1][3] = 0.0f;
		elements[2][0] = 0.0f;			elements[2][1] = 0.0f;			elements[2][2] = 1.0f;	elements[2][3] = 0.0f;
		elements[3][0] = 0.0f;			elements[3][1] = 0.0f;			elements[3][2] = 0.0f;	elements[3][3] = 1.0f;
	}



	float CMatrix::operator () (int row, int col) const
	{
		return elements[row][col];
	}



	float & CMatrix::operator () (int row, int col)
	{
		return elements[row][col];
	}



	CMatrix & CMatrix::operator = (const CMatrix &m)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				elements[i][j] = m.elements[i][j];
			}
		}

		return *this;
	}



	bool CMatrix::operator == (const CMatrix &m) const
	{
		bool equals = true;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				if ( fabs(elements[i][j] - m.elements[i][j]) > epsilon )
					equals = false;
			}
		}

		return equals;
	}



	bool CMatrix::operator != (const CMatrix &m) const
	{
		return !(*this == m);
	}



	CMatrix CMatrix::operator * (const CMatrix &m) const
	{
		CMatrix temp;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				temp.elements[i][j] =
					elements[i][0]*m.elements[0][j] +
					elements[i][1]*m.elements[1][j] +
					elements[i][2]*m.elements[2][j] +
					elements[i][3]*m.elements[3][j];
			}
		}

		return temp;
	}



	CMatrix & CMatrix::operator *= (const CMatrix &m)
	{
		CMatrix temp;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				temp.elements[i][j] =
					elements[i][0]*m.elements[0][j] +
					elements[i][1]*m.elements[1][j] +
					elements[i][2]*m.elements[2][j] +
					elements[i][3]*m.elements[3][j];
			}
		}

		*this = temp;
		return *this;
	}



	CVector3 CMatrix::operator * (const CVector3 &v) const
	{
		CVector3 temp;

		temp.x = elements[0][0]*v.x + elements[0][1]*v.y + elements[0][2]*v.z;
		temp.y = elements[1][0]*v.x + elements[1][1]*v.y + elements[1][2]*v.z;
		temp.z = elements[2][0]*v.x + elements[2][1]*v.y + elements[2][2]*v.z;

		return temp;
	}



	CVector4 CMatrix::operator * (const CVector4 &v) const
	{
		CVector4 temp;

		temp.x = elements[0][0]*v.x + elements[0][1]*v.y + elements[0][2]*v.z + elements[0][3];
		temp.y = elements[1][0]*v.x + elements[1][1]*v.y + elements[1][2]*v.z + elements[1][3];
		temp.z = elements[2][0]*v.x + elements[2][1]*v.y + elements[2][2]*v.z + elements[2][3];
		temp.w = elements[3][0]*v.x + elements[3][1]*v.y + elements[3][2]*v.z + elements[3][3];

		float oneOverW = 1.0f/temp.w;
		temp.x *= oneOverW;
		temp.y *= oneOverW;
		temp.z *= oneOverW;
		temp.w = 1.0f;

		return temp;
	}
}
