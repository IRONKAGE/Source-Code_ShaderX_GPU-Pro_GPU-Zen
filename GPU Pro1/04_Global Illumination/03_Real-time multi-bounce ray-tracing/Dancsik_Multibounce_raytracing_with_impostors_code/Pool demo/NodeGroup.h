#pragma once
#include <vector>
#include "Node.h"

//class Node;

class NodeGroup : public Node
{
	std::vector<Node*> subnodes;
public:
	NodeGroup(void);
public:
	~NodeGroup(void);

	void add(Node* e);

	virtual void render(const RenderContext& context);
	virtual void animate(double dt);
	virtual void control(double dt, Node* others);
	virtual void interact(Entity* target);
	virtual void handleMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	void affect(Entity* affector);

	virtual HRESULT createDefaultResources(EngineCore* core);
	virtual HRESULT releaseDefaultResources();
};
