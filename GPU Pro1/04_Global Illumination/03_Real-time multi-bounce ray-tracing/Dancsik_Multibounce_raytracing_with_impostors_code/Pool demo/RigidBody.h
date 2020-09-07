#pragma once
#include "entity.h"

class RigidModel;

class RigidBody : public Entity
{
	EngineCore* core;

	RigidModel* rigidModel;	//Physics model reference.

	// Orientation.
	D3DXQUATERNION orientation;
	// Momentum.
	D3DXVECTOR3 momentum;
	/// Angular momentum.
	D3DXVECTOR3 angularMomentum;

	// Force.
	D3DXVECTOR3 force;
	// Torque.
	D3DXVECTOR3 torque;

	// Auxiliary roataion matrix computed from orientation.
	D3DXMATRIX rotationMatrix;

	public:
	// Constructor.
	RigidBody(ShadedMesh* shadedMesh, RigidModel* rigidModel);

	virtual void render(const RenderContext& context);
	virtual void animate(double dt);
	virtual void control(double dt, Node* others);
	virtual HRESULT createDefaultResources(EngineCore* core);

	void setAngularMomentum(const D3DXVECTOR3& am);

	/// Returns model matrix to be used by lights and camera attached to the entity.
	virtual void getModelMatrix(D3DXMATRIX& modelMatrix);

	/// Returns the inverse of the model matrix.
	virtual void getModelMatrixInverse(D3DXMATRIX& modelMatrixInverse);

	/// Returns the inverse of the rotation matrix.
	virtual void getRotationMatrixInverse(D3DXMATRIX& rotationMatrixInverse);

	void getWorldInvMassMatrix(D3DXMATRIX& wim);

	void setForces(D3DXVECTOR3 buoyancy, float density);
	void initMomentums();
	void setPositionConstraints();
};
