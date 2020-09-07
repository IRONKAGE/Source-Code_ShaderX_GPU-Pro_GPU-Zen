#pragma once
#include "Node.h"
#include "ShadedMesh.h"
class ShadedMesh;
class NodeGroup;

class Entity : public Node
{
protected:
	ShadedMesh* shadedMesh;
	D3DXVECTOR3 position;
	
public:
	Entity(ShadedMesh* shadedMesh);

	virtual ShadedMesh* getMesh() {return shadedMesh;}
	
	virtual ~Entity(void);

	virtual void render(const RenderContext& context);
	virtual void animate(double dt);
	virtual void control(double dt, Node* others);
	virtual void interact(Entity* target);
	virtual void affect(Entity* affector);
	virtual void handleMessage(HWND hWnd,  UINT uMsg, WPARAM wParam, LPARAM lParam) {}

	virtual void setUpCamera(YawPitchRollCamera& camera);

	virtual void setPosition(const D3DXVECTOR3& position);
	virtual D3DXVECTOR3 getPosition() {return position;}

	virtual void getModelMatrix(D3DXMATRIX ModelMatrix);

	virtual HRESULT createDefaultResources(EngineCore* core);
	virtual HRESULT releaseDefaultResources();

	
};
