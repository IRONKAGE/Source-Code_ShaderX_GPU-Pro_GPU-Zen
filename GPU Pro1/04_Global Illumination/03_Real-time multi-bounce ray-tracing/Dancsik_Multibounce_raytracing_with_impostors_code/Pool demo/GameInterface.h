#pragma once

class GameInterface
{
protected:
	LPDIRECT3DDEVICE9 device;
public:
	GameInterface(LPDIRECT3DDEVICE9 device){this->device = device;}
	virtual ~GameInterface(){}
	virtual HRESULT createManagedResources(){return S_OK;}
	virtual HRESULT createDefaultResources(wchar_t* effectFileName){return S_OK;}
	virtual HRESULT releaseManagedResources(){return S_OK;}
	virtual HRESULT releaseDefaultResources(){return S_OK;}

	virtual void animate(double dt, double t){}
	virtual void processMessage( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam){}

	virtual void render(){}
};
