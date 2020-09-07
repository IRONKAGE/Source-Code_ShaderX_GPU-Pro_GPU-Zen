//--------------------------------------------------------------------------------------
// File: Sphere.h
//
// Simple struct of bounding sphere
//
// Copyright (c) Hung-Chien Liao. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

// enum for half space collision result
enum CollisionResult {
	BackSide = -1,		// In the back side
	Intersect = 0,		// Intersect between these two objects
	FrontSide = 1,		// In the front side
	Coplane = 2,		// These two planes/triangles are co-plane
	Inside,				// One object is inside the other one
	Outside,			// There is no collision
	Parallel			// Parallel
};

struct CSphere
{
	D3DXVECTOR3	m_vCenter;	// The center point of this sphere
	float		m_fRadius;	// The radius of this sphere
	bool Collision(const CSphere& test) const;

	// Sphere to plane collision detection
	CollisionResult Collision(const D3DXPLANE& pPlane) const;

	// Sphere to frustum collision detection
	CollisionResult Collision(const D3DXPLANE* testFrustum, int numPlanes);
};