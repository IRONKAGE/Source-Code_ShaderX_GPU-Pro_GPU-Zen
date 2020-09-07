#include "dxstdafx.h"
#include "DXUT.h"
#include "Spotlight.h"
#include "Entity.h"
#include "RenderContext.h"
#include "NodeGroup.h"

SpotLight::SpotLight(const D3DXVECTOR3& peakRadiance, const D3DXVECTOR3& position, const D3DXVECTOR3& direction, double focus)
{
	this->peakRadiance = peakRadiance;
	this->position = position;
	this->direction = direction;
	this->focus = focus;
	this->owner = NULL;

}


void SpotLight::setOwner(Entity* owner)
{
	this->owner = owner;
}

const D3DXVECTOR3& SpotLight::getPeakRadiance()
{
	return peakRadiance;
}

D3DXVECTOR3 SpotLight::getPosition()
{
	if(owner == NULL)
		return position;
	D3DXMATRIX ownerModelMatrix;
	owner->getModelMatrix(ownerModelMatrix);

	D3DXVECTOR3 worldPosition;
	D3DXVec3TransformCoord(&worldPosition, &position, &ownerModelMatrix);

	return worldPosition;
}

D3DXVECTOR3 SpotLight::getDirection()
{
	if(owner == NULL)
		return direction;
	D3DXMATRIX ownerModelMatrix;
	owner->getModelMatrix(ownerModelMatrix);

	D3DXVECTOR3 worldDirection;
	D3DXVec3TransformNormal(&worldDirection, &direction, &ownerModelMatrix);

	return worldDirection;
}

float SpotLight::getFocus()
{
	return focus;
}
