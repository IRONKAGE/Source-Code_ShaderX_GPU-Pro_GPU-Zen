/* $Id: math_common.cpp 247 2009-09-08 08:57:56Z chomzee $ */

#include <cmath>

#include "math_common.h"
#include "vector.h"

// ----------------------------------------------------------------------------

namespace Blossom
{
	float saturate(float value)
	{
		if (value > 1.0f)
			return 1.0f;
		else if (value < 0.0f)
			return 0.0f;
		else
			return value;
	}



    float deg2rad(float degrees)
    {
        return (degrees * (PI / 180.0f));
    }



    float rad2deg(float radians)
    {
        return (radians * (180.0f / PI));
    }



    float getDistanceBetweenPoints(const CVector2 &v1, const CVector2 &v2)
    {
        float dx = v1.x - v2.x;
        float dy = v1.y - v2.y;

        return sqrtf(dx*dx + dy*dy);
    }



    float getAngleBetweenVectors(const CVector2 &v1, const CVector2 &v2)
    {
        return acosf( (v1 % v2) / (!v1 * !v2) );
    }



    CVector2 getReflectedVector(CVector2 input, CVector2 normal)
    {
        input.normalize();
        normal.normalize();

        return (input - 2*normal * (input % normal));
    }



    float getDistanceBetweenPoints(const CVector3 &v1, const CVector3 &v2)
    {
        float dx = v1.x - v2.x;
        float dy = v1.y - v2.y;
        float dz = v1.z - v2.z;

        return sqrtf(dx*dx + dy*dy + dz*dz);
    }



    float getAngleBetweenVectors(const CVector3 &v1, const CVector3 &v2)
    {
        return acosf( (v1 % v2) / (!v1 * !v2) );
    }



    CVector3 getReflectedVector(CVector3 input, CVector3 normal)
    {
        input.normalize();
        normal.normalize();

        return (input - 2*normal * (input % normal));
    }



	// algorithm comes from Fernando's and Kilgard's "The Cg Tutorial"
	void computeTangentBasisForTriangle(
		const CVector3 &v1, const CVector2 &uv1,
		const CVector3 &v2, const CVector2 &uv2,
		const CVector3 &v3, const CVector2 &uv3,
		CVector3 &tangent, CVector3 &bitangent, CVector3 &normal)
	{
		CVector3 delta1_xuv = CVector3(v2.x - v1.x, uv2.x - uv1.x, uv2.y - uv1.y);
		CVector3 delta2_xuv = CVector3(v3.x - v1.x, uv3.x - uv1.x, uv3.y - uv1.y);
		CVector3 cross_xuv = delta1_xuv ^ delta2_xuv;

		CVector3 delta1_yuv = CVector3(v2.y - v1.y, uv2.x - uv1.x, uv2.y - uv1.y);
		CVector3 delta2_yuv = CVector3(v3.y - v1.y, uv3.x - uv1.x, uv3.y - uv1.y);
		CVector3 cross_yuv = delta1_yuv ^ delta2_yuv;

		CVector3 delta1_zuv = CVector3(v2.z - v1.z, uv2.x - uv1.x, uv2.y - uv1.y);
		CVector3 delta2_zuv = CVector3(v3.z - v1.z, uv3.x - uv1.x, uv3.y - uv1.y);
		CVector3 cross_zuv = delta1_zuv ^ delta2_zuv;

		tangent.x = - cross_xuv.y / cross_xuv.x;
		tangent.y = - cross_yuv.y / cross_yuv.x;
		tangent.z = - cross_zuv.y / cross_zuv.x;

		bitangent.x = - cross_xuv.z / cross_xuv.x;
		bitangent.y = - cross_yuv.z / cross_yuv.x;
		bitangent.z = - cross_zuv.z / cross_zuv.x;

		normal = (v2 - v1) ^ (v3 - v1);

		tangent.normalize();
		bitangent.normalize();
		normal.normalize();
	}
}
