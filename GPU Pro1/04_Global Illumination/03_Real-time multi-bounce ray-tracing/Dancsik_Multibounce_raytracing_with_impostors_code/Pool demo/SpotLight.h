#pragma once


class Entity;
class NodeGroup;
class RenderContext;

/// Light source class for point-like, directed light sources.
class SpotLight
{
protected:
	/// Entity the light is moving with. Can be NULL for a fixed light.
	Entity* owner;
	/// Peak radiance of light source, emitted in main direction.
	D3DXVECTOR3 peakRadiance;
	/// Light source position, relative to owner entity. In world space, if owner is NULL.
	D3DXVECTOR3 position;
	/// Main direction, at which the light source emits its peak radiance, relative to owner entity. In world space, if owner is NULL.
	D3DXVECTOR3 direction;
	/// Exponent for radiance falloff at directions further away from main direction.
	double focus;


public:
	/// Constructor.
	SpotLight(const D3DXVECTOR3& peakRadiance, const D3DXVECTOR3& position, const D3DXVECTOR3& direction, double focus);
	/// Assigns owner entity to light.
	void setOwner(Entity* owner);
	/// Returns peak radiance.
	const D3DXVECTOR3& getPeakRadiance();
	/// Returns world space position.
	D3DXVECTOR3 getPosition();
	/// Returns world space direction.
	D3DXVECTOR3 getDirection();
	/// Returns falloff exponent.
	float getFocus();
};