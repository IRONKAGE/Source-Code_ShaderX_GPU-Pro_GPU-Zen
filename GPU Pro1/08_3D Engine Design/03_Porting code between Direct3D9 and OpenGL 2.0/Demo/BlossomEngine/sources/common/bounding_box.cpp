/* $Id: bounding_box.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#include "bounding_box.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	void CBoundingBox::getCorners(CVector3 corners[8])
	{
		corners[0].x = min.x;
		corners[0].y = min.y;
		corners[0].z = max.z;

		corners[1].x = max.x;
		corners[1].y = min.y;
		corners[1].z = max.z;

		corners[2].x = min.x;
		corners[2].y = max.y;
		corners[2].z = max.z;

		corners[3].x = max.x;
		corners[3].y = max.y;
		corners[3].z = max.z;

		corners[4].x = min.x;
		corners[4].y = min.y;
		corners[4].z = min.z;

		corners[5].x = max.x;
		corners[5].y = min.y;
		corners[5].z = min.z;

		corners[6].x = min.x;
		corners[6].y = max.y;
		corners[6].z = min.z;

		corners[7].x = max.x;
		corners[7].y = max.y;
		corners[7].z = min.z;
	}



	void CBoundingBox::getCorners(CVector4 corners[8])
	{
		corners[0].x = min.x;
		corners[0].y = min.y;
		corners[0].z = max.z;
		corners[0].w = 1.0f;

		corners[1].x = max.x;
		corners[1].y = min.y;
		corners[1].z = max.z;
		corners[1].w = 1.0f;

		corners[2].x = min.x;
		corners[2].y = max.y;
		corners[2].z = max.z;
		corners[2].w = 1.0f;

		corners[3].x = max.x;
		corners[3].y = max.y;
		corners[3].z = max.z;
		corners[3].w = 1.0f;

		corners[4].x = min.x;
		corners[4].y = min.y;
		corners[4].z = min.z;
		corners[4].w = 1.0f;

		corners[5].x = max.x;
		corners[5].y = min.y;
		corners[5].z = min.z;
		corners[5].w = 1.0f;

		corners[6].x = min.x;
		corners[6].y = max.y;
		corners[6].z = min.z;
		corners[6].w = 1.0f;

		corners[7].x = max.x;
		corners[7].y = max.y;
		corners[7].z = min.z;
		corners[7].w = 1.0f;
	}



	void CBoundingBox::updateWithCorners(const CVector3 corners[8])
	{
		min = corners[0];
		max = corners[0];

		for (int i = 1; i < 8; i++)
		{
			if (corners[i].x < min.x)
				min.x = corners[i].x;
			if (corners[i].x > max.x)
				max.x = corners[i].x;

			if (corners[i].y < min.y)
				min.y = corners[i].y;
			if (corners[i].y > max.y)
				max.y = corners[i].y;

			if (corners[i].z < min.z)
				min.z = corners[i].z;
			if (corners[i].z > max.z)
				max.z = corners[i].z;
		}
	}



	void CBoundingBox::updateWithCorners(const CVector4 corners[8])
	{
		min = corners[0];
		max = corners[0];

		for (int i = 1; i < 8; i++)
		{
			if (corners[i].x < min.x)
				min.x = corners[i].x;
			if (corners[i].x > max.x)
				max.x = corners[i].x;

			if (corners[i].y < min.y)
				min.y = corners[i].y;
			if (corners[i].y > max.y)
				max.y = corners[i].y;

			if (corners[i].z < min.z)
				min.z = corners[i].z;
			if (corners[i].z > max.z)
				max.z = corners[i].z;
		}
	}



	void CBoundingBox::transform(const CMatrix &transform)
	{
		CVector3 corners[8];

		getCorners(corners);
		for (int i = 0; i < 8; i++)
			corners[i] = CVector4(corners[i].x, corners[i].y, corners[i].z, 1.0f) * transform;
		updateWithCorners(corners);
	}



	bool CBoundingBox::collideWithSetOfPoints(const CVector3 points[], int pointsNum)
	{
		int i;

		// x-axis

		for (i = 0; i < pointsNum; i++)
		{
			if (points[i].x < max.x)
				break;
		}
		if (i == pointsNum)
			return false;

		for (i = 0; i < pointsNum; i++)
		{
			if (points[i].x > min.x)
				break;
		}
		if (i == pointsNum)
			return false;

		// y-axis

		for (i = 0; i < pointsNum; i++)
		{
			if (points[i].y < max.y)
				break;
		}
		if (i == pointsNum)
			return false;

		for (i = 0; i < pointsNum; i++)
		{
			if (points[i].y > min.y)
				break;
		}
		if (i == pointsNum)
			return false;

		// z-axis

		for (i = 0; i < pointsNum; i++)
		{
			if (points[i].z < max.z)
				break;
		}
		if (i == pointsNum)
			return false;

		for (i = 0; i < pointsNum; i++)
		{
			if (points[i].z > min.z)
				break;
		}
		if (i == pointsNum)
			return false;

		// if we got here, then set of points must intersect or be inside the bounding box

		return true;
	}
}
