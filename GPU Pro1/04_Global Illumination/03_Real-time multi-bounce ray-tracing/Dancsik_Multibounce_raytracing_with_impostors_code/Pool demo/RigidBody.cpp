#include "dxstdafx.h"
#include "RigidBody.h"
#include "RigidModel.h"
#include "ShadedMesh.h"

RigidBody::RigidBody(ShadedMesh* shadedMesh, RigidModel* rigidModel) : Entity(shadedMesh)
{
	core = NULL;

	this->rigidModel = rigidModel;
	position = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	orientation = D3DXQUATERNION(0.0f, 0.0f, 0.0f, 1.0f);

	momentum = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	angularMomentum = D3DXVECTOR3(0.0f, 0.0f, 0.0f);

	force = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	torque = D3DXVECTOR3(0.0f, 0.0f, 0.0f);

	D3DXMatrixRotationQuaternion(&rotationMatrix, &orientation);
}

void RigidBody::render(const RenderContext& context)
{
	D3DXMATRIX positionMatrix;
	D3DXMatrixTranslation(&positionMatrix, position.x, position.y, position.z);

	D3DXMATRIX bodyModelMatrix = rotationMatrix * positionMatrix;
	D3DXMATRIX bodyModelMatrixInverse;
	D3DXMatrixInverse(&bodyModelMatrixInverse, NULL, &bodyModelMatrix);

	context.effect->SetMatrix("modelMatrix", &bodyModelMatrix);
	context.effect->SetMatrix("modelMatrixInverse", &bodyModelMatrixInverse);

	D3DXMATRIX modelViewProjMatrix = bodyModelMatrix * *context.viewMatrix * *context.projMatrix;
	context.effect->SetMatrix("modelViewProjMatrix", &modelViewProjMatrix);

	D3DXMATRIX modelViewMatrix = bodyModelMatrix * *context.viewMatrix;
	context.effect->SetMatrix("modelViewMatrix", &modelViewMatrix);

	shadedMesh->render(context);
}

void RigidBody::animate(double dt)
{
	momentum += force * dt;
	D3DXVECTOR3 velocity = momentum * rigidModel->invMass;
	position += velocity * dt;

	setPositionConstraints();

	angularMomentum += torque * dt;

	// compute inverse mass matrix
	D3DXMATRIX worldSpaceInvMassMatrix;
	getWorldInvMassMatrix(worldSpaceInvMassMatrix);
	
	// compute angular velocity vector
	D3DXVECTOR3 angularVelocity;
	D3DXVec3TransformCoord(&angularVelocity, &angularMomentum, &worldSpaceInvMassMatrix);

	// compute rotation happening in dt time
	float rotationsPerSecond = D3DXVec3Length(&angularVelocity);
	D3DXQUATERNION angularDifferenceQuaternion;
	D3DXQuaternionRotationAxis(&angularDifferenceQuaternion, &angularVelocity, rotationsPerSecond * 6.28 * dt);

	// append rotation to orientation
	orientation *= angularDifferenceQuaternion;
	D3DXMatrixRotationQuaternion(&rotationMatrix, &orientation);
}

void RigidBody::getWorldInvMassMatrix(D3DXMATRIX& wim)
{
	D3DXMATRIX transposedRotationMatrix;
	D3DXMatrixTranspose(&transposedRotationMatrix, &rotationMatrix);
	wim = transposedRotationMatrix * rigidModel->invAngularMass * rotationMatrix;
}

void RigidBody::control(double dt, Node* others)
{

}

void RigidBody::setAngularMomentum(const D3DXVECTOR3& am)
{
	this->angularMomentum = am;
}

void RigidBody::getModelMatrix(D3DXMATRIX& modelMatrix)
{
	D3DXMATRIX positionMatrix;
	D3DXMatrixTranslation(&positionMatrix, position.x, position.y, position.z);

	modelMatrix = rotationMatrix * positionMatrix;
}

void RigidBody::getModelMatrixInverse(D3DXMATRIX& modelMatrixInverse)
{
	D3DXMATRIX positionMatrix, rotationMatrixTransposed;
	D3DXMatrixTranslation(&positionMatrix, -position.x, -position.y, -position.z);
	D3DXMatrixTranspose(&rotationMatrixTransposed, &rotationMatrix);

	modelMatrixInverse = positionMatrix * rotationMatrixTransposed;
}

void RigidBody::getRotationMatrixInverse(D3DXMATRIX& rotationMatrixInverse)
{
	D3DXMatrixTranspose(&rotationMatrixInverse, &rotationMatrix);
}

void RigidBody::setForces(D3DXVECTOR3 buoyancy, float density)
{
	force = D3DXVECTOR3(0.0, 0.0, 0.0);
	torque = D3DXVECTOR3(0.0, 0.0, 0.0);

	if(rigidModel->invMass > 0.0)
	{
		force += D3DXVECTOR3(0.0, -10.0f / rigidModel->invMass, 0.0);	//gravity
		force += buoyancy;
		
		float radius = max( D3DXVec3Length(&shadedMesh->boundingBoxMin), D3DXVec3Length(&shadedMesh->boundingBoxMax));
		D3DXVECTOR3 velocity = momentum * rigidModel->invMass;
		
		force += -7.065f*radius*radius*density*velocity;		//drag

		D3DXVec3Cross(&torque, &rigidModel->getCentreOfMass(), &force);
	}
}

void RigidBody::initMomentums()
{
	momentum = D3DXVECTOR3(0,0,0);
	angularMomentum = D3DXVECTOR3(0,0,0);
	force = D3DXVECTOR3(0,0,0);
	torque = D3DXVECTOR3(0,0,0);
}

void RigidBody::setPositionConstraints()
{
	Entity* pool = core->getEntity(L"poolEntity");
	D3DXVECTOR3 bMin = pool->getMesh()->boundingBoxMin;
	D3DXVECTOR3 bMax = pool->getMesh()->boundingBoxMax;

	float scale = 0.98f;
	
	float poolRadius = scale * min( abs(bMin.x), min( abs(bMin.z), min( abs(bMax.x), abs(bMax.z) ) ) );
	float rigidBodyRadius = max( D3DXVec3Length(&shadedMesh->boundingBoxMin), D3DXVec3Length(&shadedMesh->boundingBoxMax));	
	D3DXVECTOR3 actualPos = position - pool->getPosition();
	
	float diff = D3DXVec2Length(&D3DXVECTOR2(actualPos.x,actualPos.z)) - poolRadius + rigidBodyRadius;
	
	if( actualPos.y > 0 )  diff = max( diff, actualPos.y - scale*bMax.y + rigidBodyRadius );
	else if( actualPos.y < 0 )  diff = max( diff, scale*bMin.y + rigidBodyRadius - actualPos.y );

	if( diff > 0 )
	{
		D3DXVECTOR3 diffPos;
		D3DXVec3Normalize(&diffPos, &actualPos);
		actualPos -= diff*diffPos;
		position = actualPos + pool->getPosition();
	}
}

HRESULT RigidBody::createDefaultResources(EngineCore* core)
{
	this->core = core;
	shadedMesh->createDefaultResources(core);
	return S_OK;
}