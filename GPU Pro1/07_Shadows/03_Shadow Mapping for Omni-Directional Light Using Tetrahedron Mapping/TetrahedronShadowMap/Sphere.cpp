//--------------------------------------------------------------------------------------
// File: Sphere.cpp
//
// Simple struct of bounding sphere
//
// Copyright (c) Hung-Chien Liao. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "Sphere.h"

//--------------------------------------------------------------------------------------
// Public Functions
//--------------------------------------------------------------------------------------
bool CSphere::Collision(const CSphere& test) const
{
	D3DXVECTOR3 vDistance = m_vCenter - test.m_vCenter; // Distance vector
		//	between this Sphere center point and testing Sphere center point
	float fDisSquare = D3DXVec3Dot(&vDistance, &vDistance); // The distance square
		//	between this Sphere center point and testing Sphere center point

	if ( fDisSquare <= ((m_fRadius + test.m_fRadius) *
		(m_fRadius + test.m_fRadius)) )
		return true;
	return false;
}

CollisionResult CSphere::Collision(const D3DXPLANE& pPlane) const
{
	float fResult = D3DXPlaneDotCoord(&pPlane, &m_vCenter);

	if (fResult >= m_fRadius)
		return FrontSide;
	else if (fResult <= -m_fRadius)
		return BackSide;
	return Intersect;
}

CollisionResult CSphere::Collision(const D3DXPLANE* testFrustum, int numPlanes)
{
	CollisionResult finalResult = Inside;
	CollisionResult result;
	for (int u = 0; u < numPlanes; ++u)
	{
		result = Collision(testFrustum[u]);
		if (result == FrontSide)
			return Outside;
		if (result == Intersect)
			finalResult = Intersect;
	}
	return finalResult;
}