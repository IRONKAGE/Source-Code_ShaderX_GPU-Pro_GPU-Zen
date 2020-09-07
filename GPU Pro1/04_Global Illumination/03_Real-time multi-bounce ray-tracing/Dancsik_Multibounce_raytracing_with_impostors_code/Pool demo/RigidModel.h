#pragma once

class RigidModel
{

	friend class RigidBody;

	double invMass;	//Inverse of physical mass.
	D3DXMATRIX invAngularMass;	//Inverse of mass moments of inertia matrix.
	D3DXVECTOR3 centreOfMass;


	public:

	RigidModel(	double invMass, D3DXVECTOR3 centreOfMass, double invAngularMassX, double invAngularMassY, double invAngularMassZ);
	double getInvMass();	//Returns 1/mass.
	D3DXVECTOR3 getCentreOfMass();
	const D3DXMATRIX& getInvAngularMass();	//Returns the inverse of the moments of inertia matrix.

};
