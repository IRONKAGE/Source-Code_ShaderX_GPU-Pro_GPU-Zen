#include "dxstdafx.h"
#include "NodeGroup.h"

NodeGroup::NodeGroup(void)
{
}

NodeGroup::~NodeGroup(void)
{
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		delete (*i);
		i++;
	}
}

void NodeGroup::add(Node* e)
{
	subnodes.push_back(e);
}

void NodeGroup::render(const RenderContext& context)
{
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		(*i)->render(context);
		i++;
	}
}

void NodeGroup::animate(double dt)
{
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		(*i)->animate(dt);
		i++;
	}
}

void NodeGroup::control(double dt, Node* others)
{
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		(*i)->control(dt, others);
		i++;
	}
}

void NodeGroup::interact(Entity* target)
{
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		(*i)->interact(target);
		i++;
	}
}

void NodeGroup::affect(Entity* affector)
{
	// never invoked
}

HRESULT NodeGroup::createDefaultResources(EngineCore* core)
{
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		(*i)->createDefaultResources(core);
		i++;
	}

	return S_OK;
}

HRESULT NodeGroup::releaseDefaultResources()
{
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		(*i)->releaseDefaultResources();
		i++;
	}

	return S_OK;
}

void NodeGroup::handleMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	std::vector<Node*>::iterator i = subnodes.begin();
	while(i != subnodes.end())
	{
		(*i)->handleMessage(hWnd, uMsg, wParam, lParam);
		i++;
	}
}