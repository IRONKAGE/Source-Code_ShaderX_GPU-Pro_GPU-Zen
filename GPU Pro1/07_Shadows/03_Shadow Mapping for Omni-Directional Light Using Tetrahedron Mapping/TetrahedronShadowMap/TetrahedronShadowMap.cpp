//--------------------------------------------------------------------------------------
// File: TetrahedronShadowMap.cpp
//
// Starting point for Tetrahedron Shadow Map Demo
//
// Copyright (c) Hung-Chien Liao. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTsettingsdlg.h"
#include "SDKmisc.h"
#include "resource.h"
#include "PointLight.h"
#include "Obj.h"

//#define DEBUG_VS   // Uncomment this line to debug vertex shaders 
//#define DEBUG_PS   // Uncomment this line to debug pixel shaders 

#define HELPTEXTCOLOR D3DXCOLOR( 0.0f, 1.0f, 0.3f, 1.0f )

LPCWSTR g_aszMeshFile[] =
{
    L"Misc\\Room.x",
	L"Misc\\Ring1.x",
	L"Misc\\Ring2.x",
	L"Misc\\Ring3.x",
	L"Misc\\StainedLight.x",
	L"Misc\\HemiLight.x",
	L"Misc\\HemiLight.x",
	L"Misc\\HemiLight.x",
	L"Misc\\HemiLight.x"
};

const unsigned int NUM_OBJ = 109;

D3DXMATRIX g_amInitObjWorld[9] =
{
	D3DXMATRIX( 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f ),
	D3DXMATRIX( 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 70.0f, 11.0f, 95.0f, 1.0f ),
	D3DXMATRIX( 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 70.0f, 11.0f, 95.0f, 1.0f ),
	D3DXMATRIX( 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 70.0f, 11.0f, 95.0f, 1.0f ),
	D3DXMATRIX( 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 70.0f, 11.0f, 95.0f, 1.0f ),
	D3DXMATRIX( -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 45.55f, 26.292f, 126.71f, 1.0f ),
	D3DXMATRIX( -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 94.492f, 26.292f, 126.71f, 1.0f ),
	D3DXMATRIX( 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 45.583f, 26.292f, 63.3f, 1.0f ),
	D3DXMATRIX( 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 94.525f, 26.292f, 63.3f, 1.0f )
};

//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
struct CObj;
struct CSphere;
void InitializeDialogs();
bool CALLBACK IsDeviceAcceptable( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, D3DFORMAT BackBufferFormat, bool bWindowed,
                                  void* pUserContext );
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );
HRESULT CALLBACK OnCreateDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc,
                                 void* pUserContext );
HRESULT CALLBACK OnResetDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc,
                                void* pUserContext );
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext );
void CALLBACK OnFrameRender( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext );
void RenderText();
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext );
void CALLBACK KeyboardProc( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
void CALLBACK MouseProc( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown, bool bSideButton1Down,
                         bool bSideButton2Down, int nMouseWheelDelta, int xPos, int yPos, void* pUserContext );
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );
void CALLBACK OnLostDevice( void* pUserContext );
void CALLBACK OnDestroyDevice( void* pUserContext );
void CalObjsInLight(std::list<CObj*>& objsInLight, const CSphere& lightRange);

D3DVERTEXELEMENT9 g_aVertDecl[] =
{
    { 0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
    { 0, 12, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,   0 },
    { 0, 24, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
    D3DDECL_END()
};

// enum for shadow type
enum ShadowTech {
	TETRAHEDRON,		// Tetrahedron shadow map
	TETRAHEDRON_LOOKUP,	// Tetrahedron shadow map with look up cube map
	DUAL_PARABOLOID,	// Dual-paraboloid shadow map
	CUBE				// Cube shadow map
};

//-----------------------------------------------------------------------------
// Name: class CViewCamera
// Desc: A camera class derived from CFirstPersonCamera.  The arrow keys and
//       numpad keys are disabled for this type of camera.
//-----------------------------------------------------------------------------
class CViewCamera : public CFirstPersonCamera
{
protected:
    virtual D3DUtil_CameraKeys MapKey( UINT nKey )
    {
        // Provide custom mapping here.
        // Same as default mapping but disable arrow keys.
        switch( nKey )
        {
            case 'A':
                return CAM_STRAFE_LEFT;
            case 'D':
                return CAM_STRAFE_RIGHT;
            case 'W':
                return CAM_MOVE_FORWARD;
            case 'S':
                return CAM_MOVE_BACKWARD;
            case 'Q':
                return CAM_MOVE_DOWN;
            case 'E':
                return CAM_MOVE_UP;

            case VK_HOME:
                return CAM_RESET;
        }

        return CAM_UNKNOWN;
    }
};

//-----------------------------------------------------------------------------
// Name: class CLightCamera
// Desc: A camera class derived from CFirstPersonCamera.  The letter keys
//       are disabled for this type of camera.  This class is intended for use
//       by the spot light.
//-----------------------------------------------------------------------------
class CLightCamera : public CFirstPersonCamera
{
protected:
    virtual D3DUtil_CameraKeys MapKey( UINT nKey )
    {
        // Provide custom mapping here.
        // Same as default mapping but disable arrow keys.
        switch( nKey )
        {
            case VK_LEFT:
                return CAM_STRAFE_LEFT;
            case VK_RIGHT:
                return CAM_STRAFE_RIGHT;
            case VK_UP:
                return CAM_MOVE_FORWARD;
            case VK_DOWN:
                return CAM_MOVE_BACKWARD;
            case VK_PRIOR:
                return CAM_MOVE_UP;        // pgup
            case VK_NEXT:
                return CAM_MOVE_DOWN;      // pgdn

            case VK_NUMPAD4:
                return CAM_STRAFE_LEFT;
            case VK_NUMPAD6:
                return CAM_STRAFE_RIGHT;
            case VK_NUMPAD8:
                return CAM_MOVE_FORWARD;
            case VK_NUMPAD2:
                return CAM_MOVE_BACKWARD;
            case VK_NUMPAD9:
                return CAM_MOVE_UP;
            case VK_NUMPAD3:
                return CAM_MOVE_DOWN;

            case VK_HOME:
                return CAM_RESET;
        }

        return CAM_UNKNOWN;
    }
};


//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
ID3DXFont*                      g_pFont = NULL;         // Font for drawing text
ID3DXFont*                      g_pFontSmall = NULL;    // Font for drawing text
ID3DXSprite*                    g_pTextSprite = NULL;   // Sprite for batching draw text calls
ID3DXEffect*                    g_pEffect = NULL;       // D3DX effect interface
bool                            g_bShowHelp = true;     // If true, it renders the UI control text
CDXUTDialogResourceManager      g_DialogResourceManager; // manager for shared resources of dialogs
CD3DSettingsDlg                 g_SettingsDlg;          // Device settings dialog
CDXUTDialog                     g_HUD;                  // dialog for standard controls
CFirstPersonCamera              g_VCamera;              // View camera
CObj g_Obj[NUM_OBJ];         // Scene object meshes
LPDIRECT3DVERTEXDECLARATION9    g_pVertDecl = NULL;		// Vertex decl for the sample
LPDIRECT3DTEXTURE9              g_pTexDef = NULL;       // Default texture for objects
CPointLight                     g_Light[5];             // The point lights in the scene
LPDIRECT3DTEXTURE9              g_pShadowMap1 = NULL;	// Texture to which the shadow map for is rendered
LPDIRECT3DTEXTURE9              g_pShadowMap2 = NULL;	// Texture to which the shadow map for is rendered
LPDIRECT3DTEXTURE9              g_pHardwareSM1 = NULL;	// Texture to which the Hardware shadow map is rendered
LPDIRECT3DTEXTURE9              g_pHardwareSM2 = NULL;	// Texture to which the Hardware shadow map is rendered
LPDIRECT3DCUBETEXTURE9			g_pCubeShadowMap = NULL;// Texture to which the cube shadow map for is rendered
LPDIRECT3DCUBETEXTURE9			g_pCubeToTSM = NULL;	// The look up cube map.
LPDIRECT3DSURFACE9              g_pDSShadow = NULL;     // Depth-stencil buffer for rendering to shadow map
LPDIRECT3DSURFACE9				g_pDSCubeShadow = NULL;	// Depth-stencil buffer for rendering to cube shadow map
ShadowTech						g_ShadowTech = TETRAHEDRON;
bool                            g_bRightMouseDown = false;// Indicates whether right mouse button is held
bool							g_bTSMStencil = false;	// Using stencil for TSM or not
bool							g_bHardwareShadowSupport = false;
bool							g_bHardwareShadow = false;

//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN 1
#define IDC_CHANGEDEVICE     3
#define IDC_CHECKBOX         4
#define IDC_STAINEDLIGHT     5
#define IDC_HEMILIGHT1       6
#define IDC_HEMILIGHT2       7
#define IDC_HEMILIGHT3       8
#define IDC_HEMILIGHT4       9
#define IDC_SHADOWTECH       10
#define IDC_HARDWARESHADOW   11
#define IDC_TSMSTENCIL       12


//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
INT WINAPI wWinMain( HINSTANCE, HINSTANCE, LPWSTR, int )
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

    // Initialize the camera
    g_VCamera.SetScalers( 0.01f, 15.0f );
    g_VCamera.SetRotateButtons( true, false, false );

    // Set up the view parameters for the camera
    D3DXVECTOR3 vFromPt = D3DXVECTOR3( 70.0f, 11.0f, 95.0f );
    D3DXVECTOR3 vLookatPt = D3DXVECTOR3( 70.0f, 5.0f, 113.0f );
    g_VCamera.SetViewParams( &vFromPt, &vLookatPt );

    // Set the callback functions. These functions allow DXUT to notify
    // the application about device changes, user input, and windows messages.  The 
    // callbacks are optional so you need only set callbacks for events you're interested 
    // in. However, if you don't handle the device reset/lost callbacks then the sample 
    // framework won't be able to reset your device since the application must first 
    // release all device resources before resetting.  Likewise, if you don't handle the 
    // device created/destroyed callbacks then DXUT won't be able to 
    // recreate your device resources.
    DXUTSetCallbackD3D9DeviceAcceptable( IsDeviceAcceptable );
    DXUTSetCallbackD3D9DeviceCreated( OnCreateDevice );
    DXUTSetCallbackD3D9DeviceReset( OnResetDevice );
    DXUTSetCallbackD3D9FrameRender( OnFrameRender );
    DXUTSetCallbackD3D9DeviceLost( OnLostDevice );
    DXUTSetCallbackD3D9DeviceDestroyed( OnDestroyDevice );
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackKeyboard( KeyboardProc );
    DXUTSetCallbackMouse( MouseProc );
    DXUTSetCallbackFrameMove( OnFrameMove );
    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );

    InitializeDialogs();

    // Show the cursor and clip it when in full screen
    DXUTSetCursorSettings( true, true );

    // Initialize DXUT and create the desired Win32 window and Direct3D 
    // device for the application. Calling each of these functions is optional, but they
    // allow you to set several options which control the behavior of the framework.
    DXUTInit( true, true ); // Parse the command line and show msgboxes
    DXUTSetHotkeyHandling( true, true, true );  // handle the defaul hotkeys
    DXUTCreateWindow( L"TetrahedronShadowMap" );
    DXUTCreateDevice( true, 640, 480 );

    // Pass control to DXUT for handling the message pump and 
    // dispatching render calls. DXUT will call your FrameMove 
    // and FrameRender callback when there is idle time between handling window messages.
    DXUTMainLoop();

    // Perform any application-level cleanup here. Direct3D device resources are released within the
    // appropriate callback functions and therefore don't require any cleanup code here.

    return DXUTGetExitCode();
}

//--------------------------------------------------------------------------------------
// Sets up the dialogs
//--------------------------------------------------------------------------------------
void InitializeDialogs()
{
    g_SettingsDlg.Init( &g_DialogResourceManager );
    g_HUD.Init( &g_DialogResourceManager );

    g_HUD.SetCallback( OnGUIEvent ); int iY = 10;
    g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 105, iY, 125, 22 );
    g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 105, iY += 24, 125, 22, VK_F2 );	
    g_HUD.AddCheckBox( IDC_CHECKBOX, L"Display help text", 105, iY += 24, 125, 22, true, VK_F1 );
	g_HUD.AddCheckBox( IDC_STAINEDLIGHT, L"Stain Light (Space)", 105, iY += 24, 125, 22, true, VK_SPACE );
	g_HUD.AddCheckBox( IDC_HEMILIGHT1, L"Hemi Light 1 (1)", 105, iY += 24, 125, 22, false, L'1' );
	g_HUD.AddCheckBox( IDC_HEMILIGHT2, L"Hemi Light 2 (2)", 105, iY += 24, 125, 22, false, L'2' );
	g_HUD.AddCheckBox( IDC_HEMILIGHT3, L"Hemi Light 3 (3)", 105, iY += 24, 125, 22, false, L'3' );
	g_HUD.AddCheckBox( IDC_HEMILIGHT4, L"Hemi Light 4 (4)", 105, iY += 24, 125, 22, false, L'4' );
	CDXUTComboBox* pCombo;
	g_HUD.AddComboBox( IDC_SHADOWTECH, 0, iY += 24, 240, 22, L'T', false, &pCombo );
	if (pCombo)
	{
		pCombo->SetDropHeight( 50 );
		pCombo->AddItem( L"Tetrahedron Shadow Map (T)", ( LPVOID )TETRAHEDRON );
		pCombo->AddItem( L"Tetrahedron With Look UP (T)", ( LPVOID )TETRAHEDRON_LOOKUP );
        pCombo->AddItem( L"Dual-Paraboloid Shadow Map (T)", ( LPVOID )DUAL_PARABOLOID );
        pCombo->AddItem( L"Cube Shadow Map (T)", ( LPVOID )CUBE );
	}
    g_HUD.AddCheckBox( IDC_HARDWARESHADOW, L"Hardware Shadow Map (H)", 0, iY += 24, 160, 22, false, L'H' );
    g_HUD.AddCheckBox( IDC_TSMSTENCIL, L"Using Stencil for TSM (C)", 0, iY += 24, 160, 22, false, L'C' );
}


//--------------------------------------------------------------------------------------
// Called during device initialization, this code checks the device for some 
// minimum set of capabilities, and rejects those that don't pass by returning false.
//--------------------------------------------------------------------------------------
bool CALLBACK IsDeviceAcceptable( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat,
                                  D3DFORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
    // Skip backbuffer formats that don't support alpha blending
    IDirect3D9* pD3D = DXUTGetD3D9Object();
    if( FAILED( pD3D->CheckDeviceFormat( pCaps->AdapterOrdinal, pCaps->DeviceType,
                                         AdapterFormat, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
                                         D3DRTYPE_TEXTURE, BackBufferFormat ) ) )
        return false;

    // Must support pixel shader 2.0
    if( pCaps->PixelShaderVersion < D3DPS_VERSION( 2, 0 ) )
        return false;

    // need to support D3DFMT_R32F render target
    if( FAILED( pD3D->CheckDeviceFormat( pCaps->AdapterOrdinal, pCaps->DeviceType,
                                         AdapterFormat, D3DUSAGE_RENDERTARGET,
                                         D3DRTYPE_CUBETEXTURE, D3DFMT_R32F ) ) )
        return false;

    // need to support D3DFMT_A8R8G8B8 render target
    if( FAILED( pD3D->CheckDeviceFormat( pCaps->AdapterOrdinal, pCaps->DeviceType,
                                         AdapterFormat, D3DUSAGE_RENDERTARGET,
                                         D3DRTYPE_CUBETEXTURE, D3DFMT_A8R8G8B8 ) ) )
        return false;

	// Just check if the device support hardware shadow map
	if ( SUCCEEDED(pD3D->CheckDeviceFormat(pCaps->AdapterOrdinal, D3DDEVTYPE_HAL, AdapterFormat,
		D3DUSAGE_DEPTHSTENCIL, D3DRTYPE_TEXTURE, D3DFMT_D24S8)) )
		g_bHardwareShadowSupport = true;
	else
	{
		CDXUTControl* pControl = g_HUD.GetControl( IDC_HARDWARESHADOW );
		if( pControl )
			pControl->SetEnabled(false);
	}

    return true;
}

//--------------------------------------------------------------------------------------
// This callback function is called immediately before a device is created to allow the 
// application to modify the device settings. The supplied pDeviceSettings parameter 
// contains the settings that the framework has selected for the new device, and the 
// application can make any desired changes directly to this structure.  Note however that 
// DXUT will not correct invalid device settings so care must be taken 
// to return valid device settings, otherwise IDirect3D9::CreateDevice() will fail.  
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
    assert( DXUT_D3D9_DEVICE == pDeviceSettings->ver );

    HRESULT hr;
    IDirect3D9* pD3D = DXUTGetD3D9Object();
    D3DCAPS9 caps;

    V( pD3D->GetDeviceCaps( pDeviceSettings->d3d9.AdapterOrdinal,
                            pDeviceSettings->d3d9.DeviceType,
                            &caps ) );

    // Turn vsync off
    pDeviceSettings->d3d9.pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
    g_SettingsDlg.GetDialogControl()->GetComboBox( DXUTSETTINGSDLG_PRESENT_INTERVAL )->SetEnabled( false );

    // If device doesn't support HW T&L or doesn't support 1.1 vertex shaders in HW 
    // then switch to SWVP.
    if( ( caps.DevCaps & D3DDEVCAPS_HWTRANSFORMANDLIGHT ) == 0 ||
        caps.VertexShaderVersion < D3DVS_VERSION( 1, 1 ) )
    {
        pDeviceSettings->d3d9.BehaviorFlags = D3DCREATE_SOFTWARE_VERTEXPROCESSING;
    }

    // Debugging vertex shaders requires either REF or software vertex processing 
    // and debugging pixel shaders requires REF.  
#ifdef DEBUG_VS
    if( pDeviceSettings->d3d9.DeviceType != D3DDEVTYPE_REF )
    {
        pDeviceSettings->d3d9.BehaviorFlags &= ~D3DCREATE_HARDWARE_VERTEXPROCESSING;
        pDeviceSettings->d3d9.BehaviorFlags &= ~D3DCREATE_PUREDEVICE;
        pDeviceSettings->d3d9.BehaviorFlags |= D3DCREATE_SOFTWARE_VERTEXPROCESSING;
    }
#endif
#ifdef DEBUG_PS
    pDeviceSettings->d3d9.DeviceType = D3DDEVTYPE_REF;
#endif
    // For the first device created if its a REF device, optionally display a warning dialog box
    static bool s_bFirstTime = true;
    if( s_bFirstTime )
    {
        s_bFirstTime = false;
        if( pDeviceSettings->d3d9.DeviceType == D3DDEVTYPE_REF )
            DXUTDisplaySwitchingToREFWarning( pDeviceSettings->ver );
    }

    return true;
}


//--------------------------------------------------------------------------------------
// This callback function will be called immediately after the Direct3D device has been 
// created, which will happen during application initialization and windowed/full screen 
// toggles. This is the best location to create D3DPOOL_MANAGED resources since these 
// resources need to be reloaded whenever the device is destroyed. Resources created  
// here should be released in the OnDestroyDevice callback. 
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnCreateDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc,
                                 void* pUserContext )
{
    HRESULT hr;


    V_RETURN( g_DialogResourceManager.OnD3D9CreateDevice( pd3dDevice ) );
    V_RETURN( g_SettingsDlg.OnD3D9CreateDevice( pd3dDevice ) );
    // Initialize the font
    V_RETURN( D3DXCreateFont( pd3dDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET,
                              OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                              L"Arial", &g_pFont ) );
    V_RETURN( D3DXCreateFont( pd3dDevice, 12, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET,
                              OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                              L"Arial", &g_pFontSmall ) );

    // Define DEBUG_VS and/or DEBUG_PS to debug vertex and/or pixel shaders with the 
    // shader debugger. Debugging vertex shaders requires either REF or software vertex 
    // processing, and debugging pixel shaders requires REF.  The 
    // D3DXSHADER_FORCE_*_SOFTWARE_NOOPT flag improves the debug experience in the 
    // shader debugger.  It enables source level debugging, prevents instruction 
    // reordering, prevents dead code elimination, and forces the compiler to compile 
    // against the next higher available software target, which ensures that the 
    // unoptimized shaders do not exceed the shader model limitations.  Setting these 
    // flags will cause slower rendering since the shaders will be unoptimized and 
    // forced into software.  See the DirectX documentation for more information about 
    // using the shader debugger.
    DWORD dwShaderFlags = D3DXFX_NOT_CLONEABLE;

#if defined( DEBUG ) || defined( _DEBUG )
    // Set the D3DXSHADER_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3DXSHADER_DEBUG;
    #endif

#ifdef DEBUG_VS
        dwShaderFlags |= D3DXSHADER_FORCE_VS_SOFTWARE_NOOPT;
    #endif
#ifdef DEBUG_PS
        dwShaderFlags |= D3DXSHADER_FORCE_PS_SOFTWARE_NOOPT;
    #endif

    // Read the D3DX effect file
    WCHAR str[MAX_PATH];
    V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, L"TetrahedronShadowMap.fx" ) );

    // If this fails, there should be debug output as to 
    // they the .fx file failed to compile
    V_RETURN( D3DXCreateEffectFromFile( pd3dDevice, str, NULL, NULL, dwShaderFlags,
                                        NULL, &g_pEffect, NULL ) );

    // Create vertex declaration
    V_RETURN( pd3dDevice->CreateVertexDeclaration( g_aVertDecl, &g_pVertDecl ) );

    // Initialize the meshes
    for( int i = 0; i < 9; ++i )
    {
        V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, g_aszMeshFile[i] ) );
        if( FAILED( g_Obj[i].m_Mesh.Create( pd3dDevice, str ) ) )
            return DXUTERR_MEDIANOTFOUND;
        V_RETURN( g_Obj[i].m_Mesh.SetVertexDecl( pd3dDevice, g_aVertDecl ) );
        g_Obj[i].m_mWorld = g_amInitObjWorld[i];

		// Compute the bounding sphere
		BYTE *pFirstVBuffer = 0; // The first position of vertex buffer
		g_Obj[i].m_Mesh.GetMesh()->LockVertexBuffer(0, (void **)&pFirstVBuffer);
		hr = D3DXComputeBoundingSphere((D3DXVECTOR3*)pFirstVBuffer,
			g_Obj[i].m_Mesh.GetMesh()->GetNumVertices(),
			D3DXGetFVFVertexSize(g_Obj[i].m_Mesh.GetMesh()->GetFVF()),
			&g_Obj[i].m_WorldBound.m_vCenter, &g_Obj[i].m_WorldBound.m_fRadius);
		g_Obj[i].m_Mesh.GetMesh()->UnlockVertexBuffer();
		D3DXVec3TransformCoord(&g_Obj[i].m_WorldBound.m_vCenter, &g_Obj[i].m_WorldBound.m_vCenter, &g_Obj[i].m_mWorld);
    }
	D3DXMATRIX mRotation;
	D3DXMatrixRotationX(&mRotation, D3DX_PI * 0.5f);
	D3DXMatrixMultiply(&g_Obj[2].m_mWorld, &mRotation, &g_Obj[2].m_mWorld);

	int iterObj;
	for (int iY = 0; iY < 10; ++iY)
	{
		for (int iX = 0; iX < 10; ++iX)
		{
			iterObj = 9 + iX + iY * 10;
			V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, L"Misc\\1X1X1Box.x" ) );
			if( FAILED( g_Obj[iterObj].m_Mesh.Create( pd3dDevice, str ) ) )
				return DXUTERR_MEDIANOTFOUND;
			V_RETURN( g_Obj[iterObj].m_Mesh.SetVertexDecl( pd3dDevice, g_aVertDecl ) );
			D3DXMatrixTranslation(&g_Obj[iterObj].m_mWorld, 70.0f - (iX - 4.5f) * 5,
				11.0f, 95.0f - (iY - 4.5f) * 5);
			g_Obj[iterObj].m_WorldBound.m_fRadius = 0.867f;
			memcpy(&g_Obj[iterObj].m_WorldBound.m_vCenter, g_Obj[iterObj].m_mWorld.m[3], sizeof(D3DXVECTOR3));
		}
	}

	// Initialize all the point light
	for (int i = 0; i < 5; ++i)
	{
		g_Light[i].m_Light.Diffuse.r = 1.0f;
		g_Light[i].m_Light.Diffuse.g = 1.0f;
		g_Light[i].m_Light.Diffuse.b = 1.0f;
		g_Light[i].m_Light.Diffuse.a = 1.0f;
		g_Light[i].m_Light.Attenuation0 = 0.0f;
		g_Light[i].m_Light.Attenuation1 = 0.0f;
		g_Light[i].m_Light.Attenuation2 = 0.004f;
		g_Light[i].m_Light.Range = 100.0f;
		g_Light[i].m_WorldBound.m_fRadius = g_Light[i].m_Light.Range;
		memcpy(&g_Light[i].m_mWorld, &g_Obj[i + 4].m_mWorld, sizeof(D3DXMATRIX));
		memcpy(&g_Light[i].m_WorldBound.m_vCenter, g_Light[i].m_mWorld.m[3], sizeof(D3DXVECTOR3));
		g_Light[i].m_pObj = &g_Obj[i + 4];
		g_Light[i].m_bOn = false;
	}
	g_Light[0].m_bOn = true;
	g_Light[1].m_mWorld.m[3][2] -= 0.3f;	// just move the light to the center of light mesh
	g_Light[2].m_mWorld.m[3][2] -= 0.3f;
	g_Light[3].m_mWorld.m[3][2] += 0.3f;
	g_Light[4].m_mWorld.m[3][2] += 0.3f;
	

    // World transform to identity
    D3DXMATRIXA16 mIdent;
    D3DXMatrixIdentity( &mIdent );
    V_RETURN( pd3dDevice->SetTransform( D3DTS_WORLD, &mIdent ) );

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Create the stencil mask for tetrahedron shadow map using stencil buffer
//--------------------------------------------------------------------------------------
bool InitTSMStencil(IDirect3DDevice9* pd3dDevice)
{
	HRESULT hr = 0;
	// Save old viewport
    D3DVIEWPORT9 oldViewport;
    pd3dDevice->GetViewport(&oldViewport);
	LPDIRECT3DSURFACE9 pOldRT = NULL;
    V( pd3dDevice->GetRenderTarget( 0, &pOldRT ) );

	// Create the stencil buffer mask for the regular shadow map
	// ...Set render target to shadow map surfaces
    LPDIRECT3DSURFACE9 pShadowSurf;
    if( SUCCEEDED( g_pShadowMap1->GetSurfaceLevel( 0, &pShadowSurf ) ) )
    {
        pd3dDevice->SetRenderTarget( 0, pShadowSurf );
        SAFE_RELEASE( pShadowSurf );
    }
	// ...Set depth stencil
    LPDIRECT3DSURFACE9 pOldDS = NULL;
    if( SUCCEEDED( pd3dDevice->GetDepthStencilSurface( &pOldDS ) ) )
        pd3dDevice->SetDepthStencilSurface( g_pDSShadow );

	D3DXMATRIX	mTempMatrix;
	IDirect3DVertexBuffer9	*pTriVB = NULL;
	D3DXMatrixIdentity(&mTempMatrix);
	pd3dDevice->SetTransform(D3DTS_VIEW, &mTempMatrix);
	D3DXMatrixPerspectiveFovLH(&mTempMatrix, D3DX_PI * 0.5f, 1.0f, 1.0f, 100.0f);
	pd3dDevice->SetTransform(D3DTS_PROJECTION, &mTempMatrix);

	const DWORD VertexFVF = (D3DFVF_XYZ);
	pd3dDevice->SetFVF(VertexFVF);
	pd3dDevice->CreateVertexBuffer(3 * sizeof(D3DXVECTOR3), D3DUSAGE_WRITEONLY, VertexFVF, D3DPOOL_DEFAULT, &pTriVB, NULL);
	D3DXVECTOR3 *v;
	pTriVB->Lock( 0, 0, (void**)&v, 0);
	v[0].x  = -10.0f;  v[0].y  = 10.0;  v[0].z  = 10.0f;
	v[1].x  = 10.0f;  v[1].y  = 10.0f;  v[1].z  = 10.0f;
	v[2].x  = 0.0f; v[2].y  = 0.0f; v[2].z  = 10.0f;
	pTriVB->Unlock();	
	pd3dDevice->Clear(0, NULL, D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0x00FFFFFF, 1.0f, 0L);
	pd3dDevice->BeginScene();
	pd3dDevice->SetRenderState(D3DRS_COLORWRITEENABLE, 0x00000000);
	pd3dDevice->SetRenderState(D3DRS_STENCILENABLE, TRUE);
	pd3dDevice->SetRenderState(D3DRS_STENCILFUNC, D3DCMP_ALWAYS);
	pd3dDevice->SetRenderState(D3DRS_STENCILMASK, 1);
	pd3dDevice->SetRenderState(D3DRS_STENCILWRITEMASK, 1);
	pd3dDevice->SetRenderState(D3DRS_STENCILREF, 1);
	pd3dDevice->SetRenderState(D3DRS_STENCILPASS, D3DSTENCILOP_REPLACE);
	pd3dDevice->SetStreamSource(0, pTriVB, 0, sizeof(D3DXVECTOR3));
	pd3dDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0 ,1);
	pTriVB->Lock( 0, 0, (void**)&v, 0);
	v[0].x  = 0.0f;  v[0].y  = 0.0;  v[0].z  = 10.0f;
	v[1].x  = 10.0f;  v[1].y  = -10.0f;  v[1].z  = 10.0f;
	v[2].x  = -10.0f; v[2].y  = -10.0f; v[2].z  = 10.0f;
	pTriVB->Unlock();
	pd3dDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0 ,1);
	pd3dDevice->EndScene();

	// Create the stencil buffer mask for the hardware shadow map
	if (g_bHardwareShadowSupport)
	{
		// ...Set render target to shadow map surfaces
		if( SUCCEEDED( g_pShadowMap1->GetSurfaceLevel( 0, &pShadowSurf ) ) )
		{
			pd3dDevice->SetRenderTarget( 0, pShadowSurf );
			SAFE_RELEASE( pShadowSurf );
		}
		// ...Set depth stencil
		LPDIRECT3DSURFACE9 pHardwareShadowSurf;
		if( SUCCEEDED( g_pHardwareSM1->GetSurfaceLevel( 0, &pHardwareShadowSurf ) ) )
		{
			pd3dDevice->SetDepthStencilSurface(pHardwareShadowSurf);
			SAFE_RELEASE( pHardwareShadowSurf );
		}
		pd3dDevice->Clear(0, NULL, D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL,	0x00FFFFFF, 1.0f, 0L);
		pd3dDevice->BeginScene();
		pd3dDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0 , 1);
		pTriVB->Lock( 0, 0, (void**)&v, 0);
		v[0].x  = -10.0f;	v[0].y  = 10.0;		v[0].z  = 10.0f;
		v[1].x  = 10.0f;	v[1].y  = 10.0f;	v[1].z  = 10.0f;
		v[2].x  = 0.0f;		v[2].y  = 0.0f;		v[2].z  = 10.0f;
		pTriVB->Unlock();
		pd3dDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0 , 1);
		pd3dDevice->EndScene();
		pTriVB->Release();
	}

	pd3dDevice->SetRenderState(D3DRS_STENCILENABLE, FALSE);
	pd3dDevice->SetRenderState(D3DRS_COLORWRITEENABLE, 0x0000000F);
	// Set back to normal viewport and render target
    if( pOldDS )
    {
        pd3dDevice->SetDepthStencilSurface( pOldDS );
        pOldDS->Release();
    }
    pd3dDevice->SetRenderTarget( 0, pOldRT );
    SAFE_RELEASE( pOldRT );
	pd3dDevice->SetViewport(&oldViewport);
	return true;
}

//--------------------------------------------------------------------------------------
// This callback function will be called immediately after the Direct3D device has been 
// reset, which will happen after a lost device scenario. This is the best location to 
// create D3DPOOL_DEFAULT resources since these resources need to be reloaded whenever 
// the device is lost. Resources created here should be released in the OnLostDevice 
// callback. 
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnResetDevice( IDirect3DDevice9* pd3dDevice,
                                const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;

    V_RETURN( g_DialogResourceManager.OnD3D9ResetDevice() );
    V_RETURN( g_SettingsDlg.OnD3D9ResetDevice() );

    if( g_pFont )
        V_RETURN( g_pFont->OnResetDevice() );
    if( g_pFontSmall )
        V_RETURN( g_pFontSmall->OnResetDevice() );
    if( g_pEffect )
        V_RETURN( g_pEffect->OnResetDevice() );

    // Create a sprite to help batch calls when drawing many lines of text
    V_RETURN( D3DXCreateSprite( pd3dDevice, &g_pTextSprite ) );

    // Setup the camera's projection parameters
    float fAspectRatio = pBackBufferSurfaceDesc->Width / ( FLOAT )pBackBufferSurfaceDesc->Height;
    g_VCamera.SetProjParams( D3DX_PI / 4, fAspectRatio, 0.1f, 300.0f );

    // Create the default texture (used when a triangle does not use a texture)
    V_RETURN( pd3dDevice->CreateTexture( 1, 1, 1, D3DUSAGE_DYNAMIC, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &g_pTexDef,
                                         NULL ) );
    D3DLOCKED_RECT lr;
    V_RETURN( g_pTexDef->LockRect( 0, &lr, NULL, 0 ) );
    *( LPDWORD )lr.pBits = D3DCOLOR_RGBA( 255, 255, 255, 255 );
    V_RETURN( g_pTexDef->UnlockRect( 0 ) );

    // Restore the scene objects
    for( int i = 0; i < NUM_OBJ; ++i )
        V_RETURN( g_Obj[i].m_Mesh.RestoreDeviceObjects( pd3dDevice ) );

    // Restore the effect variables
    V_RETURN( g_pEffect->SetVector( "g_vLightDiffuse", ( D3DXVECTOR4* )&g_Light[0].m_Light.Diffuse ) );
    //V_RETURN( g_pEffect->SetFloat( "g_fCosTheta", cosf( g_Light[0].m_Light.Theta ) ) );

    // Create the shadow map texture
	V_RETURN( pd3dDevice->CreateTexture( CPointLight::ShadowMapSize, CPointLight::ShadowMapSize,
                                         1, D3DUSAGE_RENDERTARGET,
                                         D3DFMT_R32F,
                                         D3DPOOL_DEFAULT,
                                         &g_pShadowMap1,
                                         NULL ) );

	// Create the shadow map texture
    V_RETURN( pd3dDevice->CreateTexture( CPointLight::ShadowMapSize, CPointLight::ShadowMapSize,
                                         1, D3DUSAGE_RENDERTARGET,
                                         D3DFMT_R32F,
                                         D3DPOOL_DEFAULT,
                                         &g_pShadowMap2,
                                         NULL ) );

	if (g_bHardwareShadowSupport)
	{
		V_RETURN( pd3dDevice->CreateTexture(CPointLight::ShadowMapSize, CPointLight::ShadowMapSize, 1,
			D3DUSAGE_DEPTHSTENCIL, D3DFMT_D24S8, D3DPOOL_DEFAULT, &g_pHardwareSM1, NULL) );

		V_RETURN( pd3dDevice->CreateTexture(CPointLight::ShadowMapSize, CPointLight::ShadowMapSize, 1,
			D3DUSAGE_DEPTHSTENCIL, D3DFMT_D24S8, D3DPOOL_DEFAULT, &g_pHardwareSM2, NULL) );
	}

	V_RETURN( pd3dDevice->CreateCubeTexture( CPointLight::CubeShadowMapSize, 1, D3DUSAGE_RENDERTARGET,
		D3DFMT_R32F, D3DPOOL_DEFAULT, &g_pCubeShadowMap, NULL) );

    // Create the depth-stencil buffer to be used with the shadow map
    // We do this to ensure that the depth-stencil buffer is large
    // enough and has correct multisample type/quality when rendering
    // the shadow map.  The default depth-stencil buffer created during
    // device creation will not be large enough if the user resizes the
    // window to a very small size.  Furthermore, if the device is created
    // with multisampling, the default depth-stencil buffer will not
    // work with the shadow map texture because texture render targets
    // do not support multisample.
    DXUTDeviceSettings d3dSettings = DXUTGetDeviceSettings();
    V_RETURN( pd3dDevice->CreateDepthStencilSurface( CPointLight::ShadowMapSize,
                                                     CPointLight::ShadowMapSize,
                                                     D3DFMT_D24S8,
                                                     D3DMULTISAMPLE_NONE,
                                                     0,
                                                     TRUE,
                                                     &g_pDSShadow,
                                                     NULL ) );

	V_RETURN( pd3dDevice->CreateDepthStencilSurface(CPointLight::CubeShadowMapSize, CPointLight::CubeShadowMapSize,
		D3DFMT_D24S8, D3DMULTISAMPLE_NONE, 0, false, &g_pDSCubeShadow, NULL) );	

	InitTSMStencil(pd3dDevice);

	// Load the lookup cube map
	WCHAR str[MAX_PATH];
	V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, L"Misc\\CubeToTSMCoord.dds" ) );
	V_RETURN( D3DXCreateCubeTextureFromFile(pd3dDevice, str, &g_pCubeToTSM));

    g_HUD.SetLocation( pBackBufferSurfaceDesc->Width - 240, 0 );
    g_HUD.SetSize( 240, pBackBufferSurfaceDesc->Height );
    CDXUTControl* pControl = g_HUD.GetControl( IDC_SHADOWTECH );
    if( pControl )
        pControl->SetLocation( 0, pBackBufferSurfaceDesc->Height - 90 );
	pControl = g_HUD.GetControl( IDC_HARDWARESHADOW );
    if( pControl )
        pControl->SetLocation( 0, pBackBufferSurfaceDesc->Height - 65 );
    pControl = g_HUD.GetControl( IDC_TSMSTENCIL );
    if( pControl )
        pControl->SetLocation( 0, pBackBufferSurfaceDesc->Height - 40 );
    return S_OK;
}


//--------------------------------------------------------------------------------------
// This callback function will be called once at the beginning of every frame. This is the
// best location for your application to handle updates to the scene, but is not 
// intended to contain actual rendering calls, which should instead be placed in the 
// OnFrameRender callback.  
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
    // Update the camera's position based on user input 
    g_VCamera.FrameMove( fElapsedTime );

	// Rotate all three rings
	D3DXMATRIX mRotation;
	D3DXMatrixRotationX(&mRotation, fElapsedTime * 0.9f);
	D3DXMatrixMultiply(&g_Obj[1].m_mWorld, &mRotation, &g_Obj[1].m_mWorld);
	D3DXMatrixRotationZ(&mRotation, fElapsedTime * 0.8f);
	D3DXMatrixMultiply(&g_Obj[2].m_mWorld, &mRotation, &g_Obj[2].m_mWorld);
	D3DXMatrixRotationZ(&mRotation, fElapsedTime * 0.7f);
	D3DXMatrixMultiply(&g_Obj[3].m_mWorld, &mRotation, &g_Obj[3].m_mWorld);
}

//--------------------------------------------------------------------------------------
// This callback function will be called at the end of every frame to perform all the 
// rendering calls for the scene, and it will also be called if the window needs to be 
// repainted. After this function has returned, DXUT will call 
// IDirect3DDevice9::Present to display the contents of the next buffer in the swap chain
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameRender( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext )
{
    // If the settings dialog is being shown, then
    // render it instead of rendering the app's scene
    if( g_SettingsDlg.IsActive() )
    {
        g_SettingsDlg.OnRender( fElapsedTime );
        return;
    }

    HRESULT hr;

	V( pd3dDevice->Clear( 0L, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x000000ff, 1.0f, 0L ) );

	// Use each light to render the scene
	bool bFirstPass = true;
	for (int i = 0; i < 5; ++i)
	{
		if (!g_Light[i].m_bOn)
			continue;
		if (bFirstPass)
		{
			V( pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, false) );
			bFirstPass = false;
		}
		else
		{
			V( pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, true) );
			V( pd3dDevice->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_ONE) );
			V( pd3dDevice->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_ONE) );
		}
		
		if (g_ShadowTech == TETRAHEDRON)
			g_Light[i].RenderSceneTSM(pd3dDevice, g_Obj, NUM_OBJ);
		else if (g_ShadowTech == TETRAHEDRON_LOOKUP)
			g_Light[i].RenderSceneTSMLookUp(pd3dDevice, g_Obj, NUM_OBJ);
		else if (g_ShadowTech == DUAL_PARABOLOID)
			g_Light[i].RenderSceneDSM(pd3dDevice, g_Obj, NUM_OBJ);
		else
			g_Light[i].RenderSceneCubeSM(pd3dDevice, g_Obj, NUM_OBJ);

		g_Light[i].RenderLight(pd3dDevice);
	}

	if( SUCCEEDED( pd3dDevice->BeginScene() ) )
    {
        // Render stats and help text
        RenderText();
        // Render the UI elements
        g_HUD.OnRender( fElapsedTime );
		V( pd3dDevice->EndScene() );
	}

    g_pEffect->SetTexture( "g_txShadowFront", NULL );
}


//--------------------------------------------------------------------------------------
// Render the help and statistics text. This function uses the ID3DXFont interface for 
// efficient text rendering.
//--------------------------------------------------------------------------------------
void RenderText()
{
    // The helper object simply helps keep track of text position, and color
    // and then it calls pFont->DrawText( m_pSprite, strMsg, -1, &rc, DT_NOCLIP, m_clr );
    // If NULL is passed in as the sprite object, then it will work however the 
    // pFont->DrawText() will not be batched together.  Batching calls will improves performance.
    CDXUTTextHelper txtHelper( g_pFont, g_pTextSprite, 15 );

    // Output statistics
    txtHelper.Begin();
    txtHelper.SetInsertionPos( 5, 5 );
    txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
    txtHelper.DrawTextLine( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) ); // Show FPS
    txtHelper.DrawTextLine( DXUTGetDeviceStats() );

    txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ) );

    // Draw help
    if( g_bShowHelp )
    {
        const D3DSURFACE_DESC* pd3dsdBackBuffer = DXUTGetD3D9BackBufferSurfaceDesc();
        txtHelper.SetInsertionPos( 10, pd3dsdBackBuffer->Height - 15 * 10 );
        txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 0.75f, 0.0f, 1.0f ) );
        txtHelper.DrawTextLine( L"Controls:" );

        txtHelper.SetInsertionPos( 15, pd3dsdBackBuffer->Height - 15 * 9 );
        WCHAR text[512];
        swprintf_s(text,L"Rotate camera\nMove camera\n"
            L"Move light\n"
            L"Hidehelp\nQuit");
        txtHelper.DrawTextLine(text);

            
        txtHelper.SetInsertionPos( 265, pd3dsdBackBuffer->Height - 15 * 9 );
        txtHelper.DrawTextLine(
            L"Left drag mouse\nW,S,A,D,Q,E\n"
            L"W,S,A,D,Q,E while holding right mouse\n"
            L"F1\nESC" );
    }
    else
    {
        txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ) );
        txtHelper.DrawTextLine( L"Press F1 for help" );
    }
    txtHelper.End();
}


//--------------------------------------------------------------------------------------
// Before handling window messages, DXUT passes incoming windows 
// messages to the application through this callback function. If the application sets 
// *pbNoFurtherProcessing to TRUE, then DXUT will not process this message.
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext )
{
    // Always allow dialog resource manager calls to handle global messages
    // so GUI state is updated correctly
    *pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    if( g_SettingsDlg.IsActive() )
    {
        g_SettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
        return 0;
    }

    *pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    // Pass all windows messages to camera and dialogs so they can respond to user input
    if( WM_KEYDOWN != uMsg || g_bRightMouseDown )
        g_Light[0].HandleMessages( hWnd, uMsg, wParam, lParam );

    if( WM_KEYDOWN != uMsg || !g_bRightMouseDown )
		g_VCamera.HandleMessages( hWnd, uMsg, wParam, lParam );

    return 0;
}


//--------------------------------------------------------------------------------------
// As a convenience, DXUT inspects the incoming windows messages for
// keystroke messages and decodes the message parameters to pass relevant keyboard
// messages to the application.  The framework does not remove the underlying keystroke 
// messages, which are still passed to the application's MsgProc callback.
//--------------------------------------------------------------------------------------
void CALLBACK KeyboardProc( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
}


void CALLBACK MouseProc( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown, bool bSideButton1Down,
                         bool bSideButton2Down, int nMouseWheelDelta, int xPos, int yPos, void* pUserContext )
{
    g_bRightMouseDown = bRightButtonDown;
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
        case IDC_CHECKBOX:
        {
            CDXUTCheckBox* pCheck = ( CDXUTCheckBox* )pControl;
            g_bShowHelp = pCheck->GetChecked();
            break;
        }
		case IDC_STAINEDLIGHT:
        {
            CDXUTCheckBox* pCheck = ( CDXUTCheckBox* )pControl;
			g_Light[0].m_bOn = pCheck->GetChecked();
            break;
        }
		case IDC_HEMILIGHT1:
        {
            CDXUTCheckBox* pCheck = ( CDXUTCheckBox* )pControl;
			g_Light[1].m_bOn = pCheck->GetChecked();
            break;
        }
		case IDC_HEMILIGHT2:
        {
            CDXUTCheckBox* pCheck = ( CDXUTCheckBox* )pControl;
			g_Light[2].m_bOn = pCheck->GetChecked();
            break;
        }
		case IDC_HEMILIGHT3:
        {
            CDXUTCheckBox* pCheck = ( CDXUTCheckBox* )pControl;
			g_Light[3].m_bOn = pCheck->GetChecked();
            break;
        }
		case IDC_HEMILIGHT4:
        {
            CDXUTCheckBox* pCheck = ( CDXUTCheckBox* )pControl;
			g_Light[4].m_bOn = pCheck->GetChecked();
            break;
        }
		case IDC_SHADOWTECH:
		{
			CDXUTComboBox* pItem = ( CDXUTComboBox* )pControl;
			if (pItem)
			{
				if (pItem->GetSelectedIndex() == TETRAHEDRON)
				{
					g_ShadowTech = TETRAHEDRON;
					CDXUTControl* pControl = g_HUD.GetControl( IDC_HARDWARESHADOW );
					if( pControl )
						pControl->SetVisible(true);
					pControl = g_HUD.GetControl( IDC_TSMSTENCIL );
					if( pControl )
					{
						pControl->SetVisible(true);
						pControl->SetEnabled(true);
					}
				}
				else if (pItem->GetSelectedIndex() == TETRAHEDRON_LOOKUP)
				{
					g_ShadowTech = TETRAHEDRON_LOOKUP;
					CDXUTCheckBox* pControl = ( CDXUTCheckBox* )g_HUD.GetControl( IDC_HARDWARESHADOW );
					if( pControl )
						pControl->SetVisible(true);
					pControl = ( CDXUTCheckBox* )g_HUD.GetControl( IDC_TSMSTENCIL );
					if( pControl )
					{
						pControl->SetVisible(true);
						pControl->SetEnabled(false);
						pControl->SetChecked(true);
						g_bTSMStencil = true;	// We need to use stencil buffer for the look up texture
					}
				}
				else if (pItem->GetSelectedIndex() == DUAL_PARABOLOID)
				{
					g_ShadowTech = DUAL_PARABOLOID;
					CDXUTControl* pControl = g_HUD.GetControl( IDC_HARDWARESHADOW );
					if( pControl )
						pControl->SetVisible(true);
					pControl = g_HUD.GetControl( IDC_TSMSTENCIL );
					if( pControl )
						pControl->SetVisible(false);
				}
				else
				{
					g_ShadowTech = CUBE;
					CDXUTControl* pControl = g_HUD.GetControl( IDC_HARDWARESHADOW );
					if( pControl )
						pControl->SetVisible(false);
					pControl = g_HUD.GetControl( IDC_TSMSTENCIL );
					if( pControl )
						pControl->SetVisible(false);
				}
			}
			break;
		}
        case IDC_HARDWARESHADOW:
        {
            CDXUTCheckBox* pCheck = ( CDXUTCheckBox* )pControl;
			g_bHardwareShadow = pCheck->GetChecked();
            break;
        }
        case IDC_TSMSTENCIL:
        {
            CDXUTCheckBox* pCheck = ( CDXUTCheckBox* )pControl;
			g_bTSMStencil = pCheck->GetChecked();
            break;
        }
    }
}


//--------------------------------------------------------------------------------------
// This callback function will be called immediately after the Direct3D device has 
// entered a lost state and before IDirect3DDevice9::Reset is called. Resources created
// in the OnResetDevice callback should be released here, which generally includes all 
// D3DPOOL_DEFAULT resources. See the "Lost Devices" section of the documentation for 
// information about lost devices.
//--------------------------------------------------------------------------------------
void CALLBACK OnLostDevice( void* pUserContext )
{
    g_DialogResourceManager.OnD3D9LostDevice();
    g_SettingsDlg.OnD3D9LostDevice();
    if( g_pFont )
        g_pFont->OnLostDevice();
    if( g_pFontSmall )
        g_pFontSmall->OnLostDevice();
    if( g_pEffect )
        g_pEffect->OnLostDevice();
    SAFE_RELEASE( g_pTextSprite );

	SAFE_RELEASE( g_pCubeToTSM );
	SAFE_RELEASE( g_pDSCubeShadow );
    SAFE_RELEASE( g_pDSShadow );
	SAFE_RELEASE( g_pCubeShadowMap );
	SAFE_RELEASE( g_pHardwareSM2 );
	SAFE_RELEASE( g_pHardwareSM1 );	
    SAFE_RELEASE( g_pShadowMap2 );
	SAFE_RELEASE( g_pShadowMap1 );
    SAFE_RELEASE( g_pTexDef );

    for( int i = 0; i < NUM_OBJ; ++i )
        g_Obj[i].m_Mesh.InvalidateDeviceObjects();
}


//--------------------------------------------------------------------------------------
// This callback function will be called immediately after the Direct3D device has 
// been destroyed, which generally happens as a result of application termination or 
// windowed/full screen toggles. Resources created in the OnCreateDevice callback 
// should be released here, which generally includes all D3DPOOL_MANAGED resources. 
//--------------------------------------------------------------------------------------
void CALLBACK OnDestroyDevice( void* pUserContext )
{
    g_DialogResourceManager.OnD3D9DestroyDevice();
    g_SettingsDlg.OnD3D9DestroyDevice();
    SAFE_RELEASE( g_pEffect );
    SAFE_RELEASE( g_pFont );
    SAFE_RELEASE( g_pFontSmall );
    SAFE_RELEASE( g_pVertDecl );

    SAFE_RELEASE( g_pEffect );

    for( int i = 0; i < NUM_OBJ; ++i )
        g_Obj[i].m_Mesh.Destroy();
}
