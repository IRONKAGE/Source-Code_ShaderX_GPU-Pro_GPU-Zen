#include "dxstdafx.h"
#include "Entity.h"
#include "ShadedMesh.h"

Entity::Entity(ShadedMesh* shadedMesh) : Node()
{
	this->shadedMesh = shadedMesh;
	position = D3DXVECTOR3(0, 0, 0);
}

Entity::~Entity(void)
{
}

void Entity::render(const RenderContext& context)
{
	if( renderable )
	{
		D3DXMATRIX modelMatrix;
		D3DXMatrixTranslation(&modelMatrix, position.x, position.y, position.z);

		D3DXMATRIX modelMatrixInverse;
		D3DXMatrixInverse(&modelMatrixInverse, NULL, &modelMatrix);
		context.effect->SetMatrix("modelMatrix", &modelMatrix);
		context.effect->SetMatrix("modelMatrixInverse", &modelMatrixInverse);

		D3DXMATRIX modelViewProjMatrix = modelMatrix* *context.viewMatrix * *context.projMatrix;
		context.effect->SetMatrix("modelViewProjMatrix", &modelViewProjMatrix);

		shadedMesh->render(context);
	}
}

void Entity::animate(double dt)
{

}

void Entity::control(double dt, Node* others)
{
}

void Entity::interact(Entity* target)
{
	target->affect(this);
}

void Entity::affect(Entity* affector)
{
	if(affector == this)
		return;
	//entity - entity interaction
}


void Entity::setUpCamera(YawPitchRollCamera& camera)
{
	// set inverse model matrix as view matrix
}

void Entity::setPosition(const D3DXVECTOR3& position)
{
	this->position = position;
}

void Entity::getModelMatrix(D3DXMATRIX ModelMatrix)
{
	D3DXMatrixTranslation(&ModelMatrix, position.x, position.y, position.z);
}

HRESULT Entity::createDefaultResources(EngineCore* core)
{
	shadedMesh->createDefaultResources(core);
	return S_OK;
}
	
HRESULT Entity::releaseDefaultResources()
{
	shadedMesh->releaseDefaultResources();
	return S_OK;
}