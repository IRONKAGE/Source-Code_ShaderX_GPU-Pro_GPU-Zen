#pragma once

#include "YawPitchRollCamera.h"
#include "RenderContext.h"
#include "EngineCore.h"

class Entity;

class Node
{
public:

	bool renderable;

	Node() {renderable = true;}
	virtual ~Node(){};
	virtual void render(const RenderContext& context)=0;
	virtual void animate(double dt)=0;
	virtual void control(double dt, Node* others)=0;
	virtual void interact(Entity* target)=0;
	virtual void affect(Entity* affector)=0;
	virtual void handleMessage(HWND hWnd,  UINT uMsg, WPARAM wParam, LPARAM lParam)=0;

	virtual HRESULT createDefaultResources(EngineCore* core)=0;
	virtual HRESULT releaseDefaultResources()=0;
};
