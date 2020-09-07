/******************************************************************************

 @File         PODPlayer.cpp

 @Title        PODPlayer

 @Version      

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     OS/API Independent

 @Description  Plays POD files using the PVREngine

******************************************************************************/

#include "PODPlayer.h"

using namespace pvrengine;


/******************************************************************************/

bool PODPlayer::InitApplication()
{
	// set up console/log
	m_pConsole = ConsoleLog::ptr();

	// grab pointers to these handlers and initialise them
	m_pTimeController = TimeController::ptr();

	m_pOptionsMenu = NULL;

	// deal with command line
	int i32NumCLOptions = PVRShellGet(prefCommandLineOptNum);
	SCmdLineOpt* sCLOptions = (SCmdLineOpt*)PVRShellGet(prefCommandLineOpts);
	CPVRTString strFilename;

	PVRESParser cParser;

	bool bFoundFile = false;
	if(sCLOptions)
	{
		for(int i=0;i<i32NumCLOptions;++i)
		{
			if(!sCLOptions[i].pVal)
			{	// could be script or pod
				strFilename=sCLOptions[i].pArg;
				// determine whether a script or a POD or nothing
				CPVRTString strExtension = PVRTStringGetFileExtension(strFilename);
				if(strExtension.toLower().compare(".pvres")==0)
				{	// script file
					m_pConsole->log("Found script:%s\n", strFilename.c_str());
					cParser.setScriptFileName(strFilename);
					bFoundFile = true;
				}
				else
				{
					if(strExtension.toLower().compare(".pod")==0)
					{	// pod file
						m_pConsole->log("Found POD\n");
						cParser.setPODFileName(strFilename);
						bFoundFile = true;
					}
					else
					{
						m_pConsole->log("Unrecognised filetype.\n");
					}
				}
			}
		}
	}
	if(!bFoundFile)
	{	// no command line options so open default pvres
		CPVRTString strDefaultPVRES((char*)(PVRShellGet(prefReadPath)));
		strDefaultPVRES += "Sample.pvres";
		cParser.setScriptFileName(strDefaultPVRES);
	}

	m_PVRES = cParser.Parse();


	// sets up whether the console should write out constantly or not
	CPVRTString strLogPath = CPVRTString((char*)PVRShellGet(prefWritePath));
	m_pConsole->setOutputFile(strLogPath+="log.txt");
	m_pConsole->setStraightToFile(m_PVRES.getLogToFile());

	m_pConsole->log("PODPlayer v0.2 alpha\n\n Initialising...\n");


	CPVRTString error = cParser.getError();
	if(!error.empty())
	{
		m_pConsole->log("Couldn't parse script: %s:\n%s",m_PVRES.getScriptFileName().c_str(),error.c_str());
		PVRShellSet(prefExitMessage, m_pConsole->getLastLogLine().c_str());
		return false;
	}

	// Deal with results of script read


	// Load the scene from the .pod file into a CPVRTModelPOD object.
	if(m_Scene.ReadFromFile(m_PVRES.getPODFileName().c_str()) != PVR_SUCCESS)
	{
		m_pConsole->log("Error: couldn't open POD file: %s\n",m_PVRES.getPODFileName().c_str());
		PVRShellSet(prefExitMessage, m_pConsole->getLastLogLine().c_str());
		return false;
	}

	// The cameras are stored in the file. Check if it contains at least one.
	if(m_Scene.nNumCamera == 0)
	{
		m_bFreeCamera = true;
	}
	else
	{
		m_bFreeCamera = false;
	}
	// use camera 0 to begin with
	m_u32CurrentCameraNum = 0;

	// Ensure that all meshes use an indexed triangle list
	for(unsigned int i = 0; i < m_Scene.nNumMesh; ++i)
	{
		if(m_Scene.pMesh[i].nNumStrips || !m_Scene.pMesh[i].sFaces.pData)
		{
			m_pConsole->log("ERROR: The meshes in the scene should use an indexed triangle list\n");
			PVRShellSet(prefExitMessage, m_pConsole->getLastLogLine().c_str());
			return false;
		}
	}

	PVRShellSet(prefFSAAMode,m_PVRES.getFSAA());					// set fullscreen anti-aliasing
	PVRShellSet(prefPowerSaving,m_PVRES.getPowerSaving());			// set power saving mode
	PVRShellSet(prefHeight,m_PVRES.getHeight());					// set height of window
	PVRShellSet(prefWidth,m_PVRES.getWidth());						// set width of window
	PVRShellSet(prefPositionX,m_PVRES.getPosX());					// set horizontal position of window
	PVRShellSet(prefPositionY,m_PVRES.getPosY());					// set vertical position of window
	PVRShellSet(prefQuitAfterTime,m_PVRES.getQuitAfterTime());		// time after which PODPlayer will automatically quit
	PVRShellSet(prefQuitAfterFrame,m_PVRES.getQuitAfterFrame());	// frame after which PODplayer will automatically quit
	PVRShellSet(prefSwapInterval,m_PVRES.getVertSync()?1:0);		// set vertical sync with monitor
	PVRShellSet(prefFullScreen, m_PVRES.getFullScreen());			// set fullscreen

	m_UniformHandler.setScene(&m_Scene);
	m_UniformHandler.setLightManager(&m_LightManager);

	// Initialize variables used for the animation
	m_bOptions = false;												// don't show options at start up
	m_bOverlayOptions = false;										// don't overlay the options by default
	m_pTimeController->setNumFrames(m_Scene.nNumFrame);				// set the number of frames to animate across
	m_pTimeController->setFrame(m_PVRES.getStartFrame());			// set the frame from which to start the animation
	m_pTimeController->setAnimationSpeed(m_PVRES.getAnimationSpeed());	// set the speed with which to animate


	// set PODPlayer to initialising state
	m_i32Initialising = 1;

	m_pConsole->log("Initial setup Succeeded\n");
	return true;
}


/******************************************************************************/

bool PODPlayer::QuitApplication()
{
	// delete options
	PVRDELETE(m_pOptionsMenu);
	return true;
}

/******************************************************************************/

bool PODPlayer::RenderScene()
{
	//TODO: put this GL specific code somewhere
	int j=0;
	do
	{
		j = glGetError();
		if(j)
		{
			ConsoleLog::inst().log("GL Error: %d %s\n",j,glGetString(j));;
		}
	}while(j);

	if(m_i32Initialising)
	{	// do initialise
		return Init();
	}
	doInput();
	// Clears the color and depth buffer
	m_PVRESettings.Clear();
	doFPS();

	// are we in the options menu
	if(m_bOptions)
	{
		if(m_bOverlayOptions)
		{	// restore background colour
			do3DScene();
		}
		m_Print3D.DisplayDefaultTitle("PODPlayer Options", "", ePVRTPrint3DLogoIMG);
		m_pOptionsMenu->render();
	}
	else
	{
		do3DScene();
		if(m_PVRES.getShowFPS())
		{
			m_Print3D.Print3D(1.0f, 15.0f, 0.5f, 0xFFff99aa, "FPS:%.1f",m_pTimeController->getFPS());
		}

		// Displays the demo name using the tools. For a detailed explanation, see the training course IntroducingPVRTools
		m_Print3D.DisplayDefaultTitle(m_strDemoTitle.c_str(), "", ePVRTPrint3DLogoIMG);
	}
	m_Print3D.Flush();
	return true;
}

/******************************************************************************/

bool PODPlayer::ReleaseView()
{
	m_pConsole->log("Exiting...\n\n");

	// Release Print3D Textures
	m_Print3D.ReleaseTextures();

	return true;
}

/*!****************************************************************************
@Function		doInput
@Description	Deals with keyboard input
******************************************************************************/
void PODPlayer::doInput()
{
	bool bUpdateOptions = false;	// grab options from menu?
	if(PVRShellIsKeyPressed(PVRShellKeyNameSELECT))
	{	// switch between options screen and normal
		m_bOptions=!m_bOptions;
		bUpdateOptions = true;
	}

	if(m_bOptions)
	{	// in options screen
		if(PVRShellIsKeyPressed(PVRShellKeyNameLEFT))
		{
			m_pOptionsMenu->prevValue();
			bUpdateOptions = true;
		}
		else if(PVRShellIsKeyPressed(PVRShellKeyNameRIGHT))
		{
			m_pOptionsMenu->nextValue();
			bUpdateOptions = true;
		}

		if(PVRShellIsKeyPressed(PVRShellKeyNameUP))
		{
			m_pOptionsMenu->prevOption();
			bUpdateOptions = true;
		}
		else if(PVRShellIsKeyPressed(PVRShellKeyNameDOWN))
		{
			m_pOptionsMenu->nextOption();
			bUpdateOptions = true;
		}
	}
	else
	{	// not options screen

		if(m_bFreeCamera)
		{

			if(PVRShellIsKeyPressed(PVRShellKeyNameLEFT))
			{
				m_Camera.YawLeft();
			}
			else if(PVRShellIsKeyPressed(PVRShellKeyNameRIGHT))
			{
				m_Camera.YawRight();
			}

			if(PVRShellIsKeyPressed(PVRShellKeyNameUP))
			{
				m_Camera.PitchDown();
			}
			else if(PVRShellIsKeyPressed(PVRShellKeyNameDOWN))
			{
				m_Camera.PitchUp();
			}

			if(PVRShellIsKeyPressed(PVRShellKeyNameACTION1))
			{	// move forward
				m_Camera.MoveForward();
			}
			else if(PVRShellIsKeyPressed(PVRShellKeyNameACTION2))
			{	// move backward
				m_Camera.MoveBack();
			}

		}
		else
		{
			if(PVRShellIsKeyPressed(PVRShellKeyNameLEFT))
			{	// rewind
				m_pTimeController->rewind();
			}
			else if(PVRShellIsKeyPressed(PVRShellKeyNameRIGHT))
			{	// fast forward
				m_pTimeController->fastforward();
			}

		}
	}

	// grab settings changes from the options menu if appropriate
	if(bUpdateOptions)
		getOptions();


}

/*!****************************************************************************
@Function		doFPS
@Description	does frames per second calc, deals with advancing the animation
in general
******************************************************************************/
void PODPlayer::doFPS()
{
	/*
	Calculates the frame number to animate in a time-based manner.
	Uses the shell function PVRShellGetTime() to get the time in milliseconds.
	*/

	float fFrame = m_pTimeController->getFrame(PVRShellGetTime());

	// Sets the scene animation to this frame
	m_Scene.SetFrame(f2vt(fFrame));
	// Sets value for if the shaders need it
	m_UniformHandler.setFrame(fFrame);
}

/*!****************************************************************************
@Function		doCamera
@Description	Deals with the current POD or free camera to set up the
projection and view matrices
******************************************************************************/
void PODPlayer::doCamera()
{
	// set up the camera
	if(m_bFreeCamera)
	{	// from the free camera class
		m_Camera.updatePosition();

		// Calculates the projection matrix
		m_UniformHandler.setProjection(m_Camera.getFOV(),
			f2vt(m_Camera.getAspect()),
			m_Camera.getNear(),
			m_Camera.getFar(),
			m_bRotate);

		// build the model view matrix from the camera position, target and an up vector.
		PVRTVec3 vUp, vTo, vPosition(m_Camera.getPosition());
		m_Camera.getTargetAndUp(vTo,vUp);
		m_UniformHandler.setView(vPosition,vTo,vUp);
	}
	else
	{	// from the POD camera values
		PVRTVec3 vFrom, vTo, vUp;
		VERTTYPE fFOV;

		// get the camera position, target and field of view (fov) with GetCameraPos()
		fFOV = m_Scene.GetCamera( vFrom, vTo, vUp, m_u32CurrentCameraNum);

		// Calculates the projection matrix
		m_UniformHandler.setProjection(fFOV,
			m_Camera.getAspect(),
			m_Scene.pCamera[m_u32CurrentCameraNum].fNear,
			m_Scene.pCamera[m_u32CurrentCameraNum].fFar,
			m_bRotate);

		// build the model view matrix from the camera position, target and an up vector.
		m_UniformHandler.setView(vFrom,vTo,vUp);
	}
}

/*!****************************************************************************
@Function		do3DScene
@Description	Renders the actual POD scene according to the current settings
******************************************************************************/
void PODPlayer::do3DScene()
{

	// clears the active material for the new frame
	m_MaterialManager.ReportActiveMaterial(NULL);
	m_UniformHandler.ResetFrameUniforms();


	doCamera();

	//// deal with lights - in OGLES2 this is not necessary as it's all done through uniforms atm
	//dynamicArray<Light*> *pdaLights = m_pLightManager->getLights();
	//for(int i=0;i<pdaLights->getSize();i++)
	//{	
	//	Light* pLight = (*pdaLights)[i];
	//	pLight->shineLight(i);
	//}

	// get the opaque meshes to draw
	dynamicArray<Mesh*> *pdaMeshes = m_MeshManager.getAll();
	bool bFrustumCull = m_PVRES.getFrustumCull();	// decide whether to cull or not
	if(pdaMeshes->getSize())
	{
		m_PVRESettings.setBlend(false);
		m_PVRESettings.setDepthWrite(true);

		for(int i=0;i<pdaMeshes->getSize();i++)
		{
			// Gets the node model matrix for this frame
			Mesh *pMesh = (*pdaMeshes)[i];
			PVRTMat4 mWorld = m_Scene.GetWorldMatrix(*pMesh->getNode());
			m_UniformHandler.setWorld(mWorld);
			// if mesh is visible - frustum test for camera
			{	// draw it
				if(bFrustumCull)
				{
					if(!m_UniformHandler.isVisibleSphere(pMesh->getCentre(),pMesh->getRadius()))
					{	// the mesh is out of frame so skip drawing it
						continue;
					}
				}
				pMesh->draw();
			}
			// else forget it
		}

	}
	
	// get the transparent meshes to draw
	pdaMeshes = m_TransparentMeshManager.getAll();
	if(pdaMeshes->getSize())
	{
		//TODO depth sort
		// at the moment these are all just rendered in whatever order without depth writes
		m_PVRESettings.setBlend(true);
		m_PVRESettings.setDepthWrite(false);

		int iNumMeshes = pdaMeshes->getSize();
		for(int i=iNumMeshes-1;i>=0;i--)
		{
			// Gets the node model matrix for this frame
			Mesh *pMesh = (*pdaMeshes)[i];
			PVRTMat4 mWorld = m_Scene.GetWorldMatrix(*pMesh->getNode());
			m_UniformHandler.setWorld(mWorld);
			// if mesh is visible - frustum test for camera
			{	// draw it
				if(bFrustumCull)
				{
					if(!m_UniformHandler.isVisibleSphere(pMesh->getCentre(),pMesh->getRadius()))
					{	// the mesh is out of frame so skip drawing it
						continue;
					}
				}
				Material *pMaterial = pMesh->getMaterial();
				// POD stores all four blend function values so use setBlendFuncSeparate

				EPODBlendFunc eSourceRGB,eDestRGB,eSourceA,eDestA;
				EPODBlendOp eOpRGB, eOpA;
				PVRTVec4 v4Colour;
				pMaterial->getBlendInfo(eSourceRGB,eDestRGB,
					eSourceA,eDestA,
					eOpRGB,eOpA,v4Colour);

				m_PVRESettings.setBlendFuncSeparate(eSourceRGB,eDestRGB,
					eSourceA,eDestA);
				m_PVRESettings.setBlendOperationSeparate(eOpRGB,eOpA);
				m_PVRESettings.setBlendColour(v4Colour);
				pMesh->draw();

			}
			// else forget it
		}

	}
	
	m_PVRESettings.setBlend(false);
	m_PVRESettings.setDepthWrite(true);
}

/*!****************************************************************************
@Function		getOptions
@Description	Updates the PODPlayer settings from the options chosen in the 
OptionsMenu
******************************************************************************/
void PODPlayer::getOptions()
{
	// Overlay options??
	m_bOverlayOptions = m_pOptionsMenu->getValueBool(eOptions_OverlayOptions);
	if(m_bOverlayOptions || !m_bOptions)
	{
		m_PVRESettings.setBackColour((VERTTYPE) m_Scene.pfColourBackground[0],(VERTTYPE) m_Scene.pfColourBackground[1],(VERTTYPE) m_Scene.pfColourBackground[2]);
	}
	else
	{
		m_PVRESettings.setBackColour(c_u32MenuBackgroundColour);
	}

	// Pause
	m_pTimeController->setFreezeTime(m_pOptionsMenu->getValueBool(eOptions_Pause));

	// FPS
	m_PVRES.setShowFPS(m_pOptionsMenu->getValueBool(eOptions_FPS));

	// POD Camera
	m_u32CurrentCameraNum = m_pOptionsMenu->getValueInt(eOptions_PODCamera);

	// Do free camera
	bool bFreeCamera = m_pOptionsMenu->getValueBool(eOptions_FreeCamera);
	if(!m_bFreeCamera && bFreeCamera)
	{	// not already free but needs to be now so match view
		PVRTVec3 vFrom, vTo, vUp;
		m_Camera.setFOV(m_Scene.GetCamera( vFrom, vTo, vUp,m_u32CurrentCameraNum));

		m_Camera.setPosition(vFrom);
		m_Camera.setTarget(vTo);
		m_Camera.setNear(m_Scene.pCamera[m_u32CurrentCameraNum].fNear);
		m_Camera.setFar(m_Scene.pCamera[m_u32CurrentCameraNum].fFar);
	}
	m_bFreeCamera = bFreeCamera;

	// invert free camera up down controls
	m_Camera.setInverted(m_pOptionsMenu->getValueBool(eOptions_Invert));

	// movement speed and rotation speed
	m_Camera.setMoveSpeed(f2vt(m_pOptionsMenu->getValueFloat(eOptions_MoveSpeed)));
	m_Camera.setRotSpeed(f2vt(m_pOptionsMenu->getValueFloat(eOptions_RotateSpeed)));

	// Draw Mode
	m_MeshManager.setDrawMode((EDrawMode)m_pOptionsMenu->getValueEnum(eOptions_DrawMode));
	m_TransparentMeshManager.setDrawMode((EDrawMode)m_pOptionsMenu->getValueEnum(eOptions_DrawMode));

	// Direction of play
	m_pTimeController->setForwards(m_pOptionsMenu->getValueBool(eOptions_Direction));

	// Frame rate
	m_pTimeController->setAnimationSpeed(m_pOptionsMenu->getValueFloat(eOptions_AnimationSpeed));
}


enum INIT_STAGES{
	eINIT_BASIC = 1,
	eINIT_PRINT3D,
	eINIT_BASICSCENE,
	eINIT_LIGHTS,
	eINIT_MATERIALMANAGER,
	eINIT_MATERIALS,
	eINIT_MESHES,
	eINIT_MENU,
	eINIT_FINISH,
}; /*!* Initialisation stages */

/*!****************************************************************************
@Function		Init
@Return			bool		whether this stage of initialisation was successful
@Description	Switches through each stage of initialisation for the PODPlayer
******************************************************************************/
bool PODPlayer::Init()
{
	//m_PVRESettings.Clear();
	switch(m_i32Initialising)
	{
	case eINIT_BASIC:
		{
			// this is the best way to determine if the view is rotated.
			m_bRotate = PVRShellGet(prefIsRotated) && PVRShellGet(prefFullScreen);

			SPVRTContext *pContext = new SPVRTContext;
			ContextManager::inst().addNewContext(pContext);		// add a context
			ContextManager::inst().initContext();		// add a context
			m_UniformHandler.setContext(pContext);
			m_PVRESettings.Init();		// mandatory initialisation step
			m_PVRESettings.setClearFlags(PVR_COLOUR_BUFFER|PVR_DEPTH_BUFFER);
			m_PVRESettings.Clear();
			m_PVRESettings.setDepthTest(true);
			m_PVRESettings.setCullMode(PVR_BACK);

			m_Camera.setAspect((VERTTYPE)PVRShellGet(prefHeight),(VERTTYPE)PVRShellGet(prefWidth));


			m_pConsole->log("Basic initialisation complete\n");
			m_i32Initialising++;
		}
		break;
	case eINIT_PRINT3D:

		m_PVRESettings.InitPrint3D(m_Print3D,PVRShellGet(prefWidth),PVRShellGet(prefHeight), m_bRotate);
		m_strPrint3DString =  "Print3D initialisation complete.";
		m_pConsole->log("Print3D initialisation complete\n");
		m_i32Initialising++;
		break;
	case eINIT_BASICSCENE:
		{
			m_strDemoTitle = m_PVRESettings.getAPIName() + " " + (m_PVRES.getTitle());

			m_strPrint3DString = "Opened Scene.";
			m_pConsole->log("Opened Scene.\n");
		}
		m_i32Initialising++;
		break;
	case eINIT_MATERIALMANAGER:
		{
			//	Load the material files
			if(!m_MaterialManager.init(&m_UniformHandler))
			{
				m_pConsole->log("Error: material manager failed to initialise\n");
				PVRShellSet(prefExitMessage, m_pConsole->getLastLogLine().c_str());
				return false;
			}
		}
		m_i32Initialising++;
		m_i32InitSub = 0;
		break;
	case eINIT_MATERIALS:
		{


			if(m_i32InitSub<(int)m_Scene.nNumMaterial)
			{
				SPODTexture sTexture;
				if(m_Scene.pMaterial[m_i32InitSub].nIdxTexDiffuse>=0)
				{
					sTexture = m_Scene.pTexture[m_Scene.pMaterial[m_i32InitSub].nIdxTexDiffuse];
				}

				m_pConsole->log("Loading Material: %s\n",m_Scene.pMaterial[m_i32InitSub].pszName);
				// because POD stores the textures that may be used by a material separately from the material
				// have to retrieve and pass texture now even if it's completely ignored
				Material *psMaterial = m_MaterialManager.LoadMaterial(m_PVRES.getPFXPath(),
					m_PVRES.getTexturePath(),
					m_Scene.pMaterial[m_i32InitSub],
					sTexture,
					&m_UniformHandler
					);
				if(!psMaterial)
				{
					m_pConsole->log("Error: Material failed to initialise\n");
					PVRShellSet(prefExitMessage, m_pConsole->getLastLogLine().c_str());
					return false;	//TODO: error handle better than this
				}
				m_strPrint3DString = CPVRTString("Initialised Material: ")+CPVRTString(m_Scene.pMaterial[m_i32InitSub].pszName);
				m_i32InitSub++;
			}
			else
			{	// done all materials lets go to the next step
				m_i32Initialising++;
				m_i32InitSub=0;
				m_strPrint3DString = "Initialised Materials.";
				m_pConsole->log("Initialised materials.\n");
			}
		}
		break;
	case eINIT_MESHES:
		{
			int i32NumMeshes = (int)m_Scene.nNumMeshNode;

			if(i32NumMeshes)
			{
				if(m_i32InitSub<i32NumMeshes)
				{
					Mesh *pMesh = new Mesh(&m_Scene, m_i32InitSub, &m_MaterialManager);

					// sort meshes into opaque and transparent (assumes no alpha test atm)
					if(pMesh->getMaterial()->getBlend())
					{
						m_TransparentMeshManager.add(pMesh);
					}
					else
					{
						m_MeshManager.add(pMesh);
					}
					m_strPrint3DString = CPVRTString("Initialised mesh: ")+CPVRTString(pMesh->getNode()->pszName);
					m_pConsole->log("%s\n", m_strPrint3DString.c_str());
					m_i32InitSub++;
				}
				else
				{
					// sort the meshes in terms of materials used
					// to avoid unnecessary shader loading etc.
					m_pConsole->log("Sorting meshes...");

					m_MeshManager.sort();
					m_TransparentMeshManager.sort();
					m_pConsole->log("Sorted\n");

					m_strPrint3DString = "Initialised meshes.";
					m_pConsole->log("Initialised meshes.\n");
					m_i32Initialising++;
				}
			}
			else
			{
					m_pConsole->log("Error: No meshes found in the POD file\n");
					PVRShellSet(prefExitMessage, m_pConsole->getLastLogLine().c_str());
					return false;	//TODO: error handle better than this
			}
		}
		break;
	case eINIT_LIGHTS:
		{
			// go through lights in scene and add them to the manager
			// We check if the scene contains any lights
			if (m_Scene.nNumLight == 0 && m_PVRES.getDefaultLight())
			{
				m_pConsole->log("Warning: no lights found in scene.\n");
				m_pConsole->log("Adding a default directional light at (0,-1,0).\n");

				// add a default light

				m_LightManager.add(new LightDirectional(PVRTVec3(0.f,-1.f,0.f),PVRTVec3(1.f,1.f,1.f)));
			}
			else
			{
				for(int i=0;i<(int)m_Scene.nNumLight;i++)
				{
					// add light i from this scene to the manager

					Light *pLight;
					if(m_Scene.pLight[i].eType==ePODDirectional)
					{// make directional light

						pLight = new LightPODDirectional(&m_Scene,i);
					}
					else
					{// make point light
						pLight = new LightPODPoint(&m_Scene,i);
					}
					m_LightManager.add(pLight);
				}
			}

			m_strPrint3DString = "Initialised lights.";
			m_pConsole->log("Initialised lights.\n");
		}
		m_i32Initialising++;
		break;
	case eINIT_MENU:
		{
			// initialises options menu and sets options values
			m_pOptionsMenu = new OptionsMenu(&m_Print3D);
			m_pOptionsMenu->addOption(new OptionEnum("Overlay Options",strOnOff,2,0));
			m_pOptionsMenu->addOption(new OptionEnum("Pause",strOnOff,2,0));
			m_pOptionsMenu->addOption(new OptionEnum("Draw Mode",g_strDrawModeNames,eNumDrawModes,m_PVRES.getDrawMode()));
			m_pOptionsMenu->addOption(new OptionInt("POD Camera",0,m_Scene.nNumCamera-1,1,0));
			m_pOptionsMenu->addOption(new OptionEnum("Free Camera",strOnOff,2,m_bFreeCamera?1:0));
			m_pOptionsMenu->addOption(new OptionEnum("  Invert Up/Down", strOnOff,2,m_Camera.getInverted()));
			m_pOptionsMenu->addOption(new OptionFloat("  Movement Speed",0.5f,100.0f,2.0f,10.0f));
			m_pOptionsMenu->addOption(new OptionFloat("  Rotation Speed",0.01f,0.5f,0.05f,0.05f));
			m_pOptionsMenu->addOption(new OptionEnum("Show FPS",strOnOff,2,m_PVRES.getShowFPS()));
			m_pOptionsMenu->addOption(new OptionEnum("Play Direction",strForwardBackward,2,m_pTimeController->getForwards()));
			m_pOptionsMenu->addOption(new OptionFloat("Animation Speed",-10,10,0.2f,m_pTimeController->getAnimationSpeed()));

			getOptions();	// update settings to reflect options

			m_strPrint3DString = "Initialised menu.";
			m_pConsole->log("Initialised menu.\n");
		}
		m_i32Initialising++;
		break;

	case eINIT_FINISH:
		m_i32Initialising=0;
		m_pConsole->log("Initialisation Complete.\n");
		m_pConsole->log(" ");
		m_strPrint3DString = "Initialisation Complete.";
		// starts time controller from this moment
		m_pTimeController->start(PVRShellGetTime());
		return true;
	}
	if(m_i32Initialising>eINIT_PRINT3D)	// i.e. if print3D is initialised
	{
		m_PVRESettings.Clear();
		m_Print3D.DisplayDefaultTitle(m_strDemoTitle.c_str(), m_strPrint3DString.c_str(), ePVRTPrint3DLogoIMG);
		m_Print3D.Flush();
	}
	return true;
}



/*!****************************************************************************
@Function		NewDemo
@Return			PVRShell*		The demo supplied by the user
@Description	This function must be implemented by the user of the shell.
The user should return its PVRShell object defining the
behaviour of the application.
******************************************************************************/
PVRShell* NewDemo()
{
	return new PODPlayer();
}
/******************************************************************************
End of file (PODPlayer.cpp)
******************************************************************************/

