/*
	Virtual texture mapping demo app
    Copyright (C) 2008, 2009 Matthäus G. Chajdas
    Contact: shaderx8@anteru.net

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "DXUT.h"
#include "DXGI.h"
#include "DXUTgui.h"
#include "DXUTmisc.h"
#include "DXUTCamera.h"
#include "DXUTSettingsDlg.h"
#include "SDKmisc.h"
#include "SDKmesh.h"
#include "dxfwd.h"
#include "strsafe.h"
#include "app.h"

#include <iostream>

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
CFirstPersonCamera          g_Camera;               // A model viewing camera
CDXUTDialogResourceManager  g_DialogResourceManager; // manager for shared resources of dialogs
CD3DSettingsDlg             g_SettingsDlg;          // Device settings dialog
CDXUTTextHelper*            g_pTxtHelper = NULL;
CDXUTDialog                 g_HUD;                  // dialog for standard controls
CDXUTDialog                 g_SampleUI;             // dialog for sample specific controls

// Direct3D 10 resources
ID3DX10Font*                g_pFont10 = NULL;
ID3DX10Sprite*              g_pSprite10 = NULL;
ID3D10EffectScalarVariable* g_pfTime = NULL;

bool						g_ShowFrameTime = false;
LARGE_INTEGER				g_StartTime, g_EndTime, g_Frequency;

D3DXMATRIX					g_View;

App*						theApp = 0;

//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN    1
#define IDC_TOGGLEREF           2
#define IDC_CHANGEDEVICE        3
#define IDC_CHANGEREQUESTCOUNT  4
#define IDC_CHANGEBINDCOUNT		5
#define IDC_USEPROGRESSIVELOAD	6
#define IDC_REQUEST_COUNT		7
#define IDC_BIND_COUNT			8

//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int main( int, char** )
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

	// DXUT will create and use the best device (either D3D9 or D3D10) 
	// that is available on the system depending on which D3D callbacks are set below

	// Set DXUT callbacks
	DXUTSetCallbackMsgProc( MsgProc );
	DXUTSetCallbackKeyboard( OnKeyboard );
	DXUTSetCallbackFrameMove( OnFrameMove );
	DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );

	DXUTSetCallbackD3D10DeviceAcceptable( IsD3D10DeviceAcceptable );
	DXUTSetCallbackD3D10DeviceCreated( OnD3D10CreateDevice );
	DXUTSetCallbackD3D10SwapChainResized( OnD3D10ResizedSwapChain );
	DXUTSetCallbackD3D10SwapChainReleasing( OnD3D10ReleasingSwapChain );
	DXUTSetCallbackD3D10DeviceDestroyed( OnD3D10DestroyDevice );
	DXUTSetCallbackD3D10FrameRender( OnD3D10FrameRender );

	InitApp();

	DXUTInit( true, true, NULL ); // Parse the command line, show msgboxes on error, no extra command line params
	DXUTSetCursorSettings( true, true );
	DXUTCreateWindow( L"VTM" );
	DXUTCreateDevice( true, 1024, 768 );
	DXUTMainLoop(); // Enter into the DXUT render loop
	DXUTShutdown();

	return 0;
}


//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{
	g_SettingsDlg.Init( &g_DialogResourceManager );
	g_HUD.Init( &g_DialogResourceManager );
	g_SampleUI.Init( &g_DialogResourceManager );

	g_HUD.SetCallback( OnGUIEvent ); int iY = 10;
	g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 35, iY, 125, 22 );
	g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 35, iY += 24, 125, 22, VK_F2 );
	g_HUD.AddCheckBox (IDC_USEPROGRESSIVELOAD, L"Progressive load", 35, iY += 24, 125, 22, true);

	g_HUD.AddStatic (IDC_REQUEST_COUNT, L"Requested pages: 20", 35, iY += 24, 125, 22);
	g_HUD.AddSlider (IDC_CHANGEREQUESTCOUNT, 35, iY += 24, 125, 22, 0, 100, 20);
	g_HUD.AddStatic (IDC_BIND_COUNT, L"Bound pages: 5", 35, iY += 24, 125, 22);
	g_HUD.AddSlider (IDC_CHANGEBINDCOUNT, 35, iY += 24, 125, 22, 0, 100, 5);
	g_SampleUI.SetCallback( OnGUIEvent ); iY = 10;

	// Setup performance counter
	QueryPerformanceFrequency (&g_Frequency);
}



//--------------------------------------------------------------------------------------
// Create any D3D10 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D10CreateDevice( ID3D10Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
									 void* pUserContext )
{
	HRESULT hr;

	V_RETURN( D3DX10CreateSprite( pd3dDevice, 500, &g_pSprite10 ) );
	V_RETURN( g_DialogResourceManager.OnD3D10CreateDevice( pd3dDevice ) );
	V_RETURN( g_SettingsDlg.OnD3D10CreateDevice( pd3dDevice ) );
	V_RETURN( D3DX10CreateFont( pd3dDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET,
		OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
		L"Arial", &g_pFont10 ) );
	g_pTxtHelper = new CDXUTTextHelper( NULL, NULL, g_pFont10, g_pSprite10, 15 );

	// Setup the camera's view parameters
	D3DXVECTOR3 vecEye( 0.0f, 0.0f, -5.0f );
	D3DXVECTOR3 vecAt ( 0.0f, 0.0f, -0.0f );
	g_Camera.SetViewParams( &vecEye, &vecAt );

	if (theApp)
	{
		delete theApp;
	}

	theApp = new App (pd3dDevice);
	theApp->Init ();

	return S_OK;
}

//--------------------------------------------------------------------------------------
// Create any D3D10 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D10ResizedSwapChain( ID3D10Device* pd3dDevice, IDXGISwapChain* pSwapChain,
										 const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	HRESULT hr;

	V_RETURN( g_DialogResourceManager.OnD3D10ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
	V_RETURN( g_SettingsDlg.OnD3D10ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

	// Setup the camera's projection parameters
	float fAspectRatio = pBackBufferSurfaceDesc->Width / ( FLOAT )pBackBufferSurfaceDesc->Height;
	g_Camera.SetProjParams( D3DX_PI / 4, fAspectRatio, 0.1f, 1000.0f );
	// g_Camera.SetWindow( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );
	// g_Camera.SetButtonMasks( MOUSE_LEFT_BUTTON, MOUSE_WHEEL, MOUSE_MIDDLE_BUTTON );
	g_Camera.SetRotateButtons (true, false, false);

	g_HUD.SetLocation( pBackBufferSurfaceDesc->Width - 170, 0 );
	g_HUD.SetSize( 170, 170 );
	g_SampleUI.SetLocation( pBackBufferSurfaceDesc->Width - 170, pBackBufferSurfaceDesc->Height - 300 );
	g_SampleUI.SetSize( 170, 300 );

	theApp->OnFinishResize ();

	return S_OK;
}

//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10ReleasingSwapChain( void* pUserContext )
{
	g_DialogResourceManager.OnD3D10ReleasingSwapChain();

	theApp->OnBeginResize ();
}

//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10DestroyDevice( void* pUserContext )
{
	theApp->Shutdown ();
	delete theApp;
	theApp = 0;

	g_DialogResourceManager.OnD3D10DestroyDevice();
	g_SettingsDlg.OnD3D10DestroyDevice();
	SAFE_RELEASE( g_pFont10 );
	SAFE_RELEASE( g_pSprite10 );
	SAFE_DELETE( g_pTxtHelper );
}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D10 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10FrameRender( ID3D10Device* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext )
{
	// Query start time
	QueryPerformanceCounter (&g_EndTime);

	const long long diff = g_EndTime.QuadPart - g_StartTime.QuadPart;

	if (g_ShowFrameTime)
	{
		std::cout << static_cast<double>(diff)/(g_Frequency.QuadPart) << "\n";
	}

	QueryPerformanceCounter (&g_StartTime);

	D3DXMATRIX mView;
	D3DXMATRIX mProj;

	float ClearColor[4] = { 0.176f, 0.196f, 0.667f, 0.0f };
	ID3D10RenderTargetView* pRTV = DXUTGetD3D10RenderTargetView();
	pd3dDevice->ClearRenderTargetView( pRTV, ClearColor );

	// Clear the depth stencil
	ID3D10DepthStencilView* pDSV = DXUTGetD3D10DepthStencilView();
	pd3dDevice->ClearDepthStencilView( pDSV, D3D10_CLEAR_DEPTH, 1.0, 0 );

	// If the settings dialog is being shown, then render it instead of rendering the app's scene
	if( g_SettingsDlg.IsActive() )
	{
		g_SettingsDlg.OnRender( fElapsedTime );
		return;
	}

	mView = *g_Camera.GetViewMatrix();

	mProj = *g_Camera.GetProjMatrix();

	DXUT_BeginPerfEvent (DXUT_PERFEVENTCOLOR2, L"Scene");
	theApp->Render (mView * mProj);
	DXUT_EndPerfEvent ();  

	// Render the HUD last so it is on top
	DXUT_BeginPerfEvent( DXUT_PERFEVENTCOLOR, L"HUD / Stats" );
	RenderText();
	g_HUD.OnRender( fElapsedTime );
	g_SampleUI.OnRender( fElapsedTime );
	DXUT_EndPerfEvent();  
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene.  This is called regardless of which D3D API is used
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
	// Update the camera's position based on user input 
	g_Camera.FrameMove( fElapsedTime );
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
						 void* pUserContext )
{
	// Pass messages to dialog resource manager calls so GUI state is updated correctly
	*pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;

	// Pass messages to settings dialog if its active
	if( g_SettingsDlg.IsActive() )
	{
		g_SettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
		return 0;
	}

	// Give the dialogs a chance to handle the message first
	*pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;
	*pbNoFurtherProcessing = g_SampleUI.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;

	// Pass all remaining windows messages to camera so it can respond to user input
	g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );

	return 0;
}


//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
	if (nChar == 'T' && bKeyDown)
	{
		g_ShowFrameTime = !g_ShowFrameTime;
	}
	else if (nChar == 'Z' && bKeyDown)
	{
		theApp->ResetCache ();
	}
}


//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
	switch( nControlID )
	{
	case IDC_TOGGLEFULLSCREEN:
		DXUTToggleFullScreen(); break;
	case IDC_CHANGEDEVICE:
		g_SettingsDlg.SetActive( !g_SettingsDlg.IsActive() ); break;
	case IDC_CHANGEREQUESTCOUNT:
		{
			WCHAR sz[100];
			StringCchPrintf( sz, 100, L"Requested pages: %d", g_HUD.GetSlider (IDC_CHANGEREQUESTCOUNT)->GetValue ());	    
			g_HUD.GetStatic (IDC_REQUEST_COUNT)->SetText (sz);
			theApp->SetMaximumTileRequestsPerFrame (g_HUD.GetSlider (IDC_CHANGEREQUESTCOUNT)->GetValue ());
			break;
		}
	case IDC_USEPROGRESSIVELOAD:
		theApp->SetProgressiveLoading (g_HUD.GetCheckBox (IDC_USEPROGRESSIVELOAD)->GetChecked ());
		break;

	case IDC_CHANGEBINDCOUNT:
		{
			WCHAR sz[100];
			StringCchPrintf( sz, 100, L"Bound pages: %d", g_HUD.GetSlider (IDC_CHANGEBINDCOUNT)->GetValue ());	    
			g_HUD.GetStatic (IDC_BIND_COUNT)->SetText (sz);
			theApp->SetMaximumTilesBoundPerFrame (g_HUD.GetSlider (IDC_CHANGEBINDCOUNT)->GetValue ());
			break;
		}
	}
}


