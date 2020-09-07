#include "dxstdafx.h"
#include "RigidModel.h"

RigidModel::RigidModel(	double invMass, D3DXVECTOR3 centreOfMass, double invAngularMassX, double invAngularMassY, double invAngularMassZ)
{
	this->invMass = invMass;
	D3DXMatrixScaling(&invAngularMass, invAngularMassX, invAngularMassY, invAngularMassZ);
	this->centreOfMass = centreOfMass;
}

double RigidModel::getInvMass()
{
	return invMass;
}

const D3DXMATRIX& RigidModel::getInvAngularMass()
{
	return invAngularMass;
}

D3DXVECTOR3 RigidModel::getCentreOfMass()
{
	return centreOfMass;
}