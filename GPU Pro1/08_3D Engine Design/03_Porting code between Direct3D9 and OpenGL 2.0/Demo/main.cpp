#ifdef WIN32
	#if defined(DEBUG) | defined(_DEBUG)
		#include <crtdbg.h>
	#endif
#endif

#include "BlossomEngine/sources/blossom_engine.h"
#include <cmath>

using namespace Blossom;



CCamera camera;

CRenderTarget sceneRenderTarget, shadowMapRenderTarget;
CDepthStencilSurface sceneDepthStencilSurface, shadowMapDepthStencilSurface;

CVertexDeclaration screenQuadVertexDeclaration;
CVertexBuffer screenQuadVertexBuffer;
CIndexBuffer screenQuadIndexBuffer;

CShader screenQuadVertexShader, screenQuadPixelShader;
CShader shadowMapPassVertexShader, shadowMapPassPixelShader;
CShader lightPassVertexShader, lightPassPixelShader;

CTexture grassDiffuseMap, brickDiffuseMap;
CTexture grassNormalMap, brickNormalMap;

CMesh terrainMesh, stuffMesh;

CMatrix terrainWorldTransform, stuffWorldTransform;
CMatrix viewProjTransform;
CMatrix lightViewProjTransform;
CMatrix projTexCoordTransform;

CVector3 lightDirection;



struct ScreenQuadVertex
{
	CVector3 position;
	CVector2 texCoord0;
};



void startPhysics()
{
	camera.updateFixed(CVector3(0.0f, 100.0f, 0.0f), CVector3(0.0f, 0.0f, 0.0f));
}



void doPhysics(int deltaTime)
{
	CVector3 eye;
	float speed = 0.01f;
	if (CApplication::isKeyPressed(SDLK_LSHIFT))
		speed = 0.2f;

	eye = camera.getEye();
	if (CApplication::isKeyPressed(SDLK_w))
		eye += speed * deltaTime * camera.getForwardVector();// CVector3(camera.getForwardVector().x, 0.0f, camera.getForwardVector().z).getNormalized();
	if (CApplication::isKeyPressed(SDLK_s))
		eye -= speed * deltaTime * camera.getForwardVector();// CVector3(camera.getForwardVector().x, 0.0f, camera.getForwardVector().z).getNormalized();
	if (CApplication::isKeyPressed(SDLK_a))
		eye -= speed * deltaTime * camera.getRightVector();
	if (CApplication::isKeyPressed(SDLK_d))
		eye += speed * deltaTime * camera.getRightVector();

	camera.updateFree(eye);

/*	if (CApplication::isKeyPressed(SDLK_a))
		camera.distanceFromEyeToAt -= 15.0f;
	if (CApplication::isKeyPressed(SDLK_z))
		camera.distanceFromEyeToAt += 15.0f;
	camera.updateFocused(CVector3());
	eye = camera.getEye();
//	if (eye.x > 100.0f)
//		eye.x = 100.0f;
	camera.updateFixed(eye, camera.getAt());*/
}



void startRender(int screenWidth, int screenHeight)
{
	#ifdef RNDR_D3D
		projTexCoordTransform = CMatrix
		(
			0.5f,  0.0f, 0.0f, 0.0f,
			0.0f, -0.5f, 0.0f, 0.0f,
			0.0f,  0.0f, 1.0f, 0.0f,
			0.5f,  0.5f, 0.0f, 1.0f
		);
	#else
		projTexCoordTransform = CMatrix
		(
			0.5f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.5f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.5f, 0.0f,
			0.5f, 0.5f, 0.5f, 1.0f
		);
	#endif



	screenQuadVertexDeclaration.init(3, 0, 0, 2);
	screenQuadVertexBuffer.init(4*sizeof(ScreenQuadVertex));
	{
		ScreenQuadVertex *vertices;

		screenQuadVertexBuffer.map((void*&)vertices);
		{
			vertices[0].position = CVector3(-1.0f, 1.0f, 0.0f);
			vertices[1].position = CVector3(-1.0f, -1.0f, 0.0f);
			vertices[2].position = CVector3(1.0f, 1.0f, 0.0f);
			vertices[3].position = CVector3(1.0f, -1.0f, 0.0f);

			#ifdef RNDR_D3D
				vertices[0].texCoord0 = CVector2(0.0f, 0.0f);
				vertices[1].texCoord0 = CVector2(0.0f, 1.0f);
				vertices[2].texCoord0 = CVector2(1.0f, 0.0f);
				vertices[3].texCoord0 = CVector2(1.0f, 1.0f);
			#else
				vertices[0].texCoord0 = CVector2(0.0f, 1.0f);
				vertices[1].texCoord0 = CVector2(0.0f, 0.0f);
				vertices[2].texCoord0 = CVector2(1.0f, 1.0f);
				vertices[3].texCoord0 = CVector2(1.0f, 0.0f);
			#endif
		}
		screenQuadVertexBuffer.unmap();
	}
	screenQuadIndexBuffer.init(6*2);
	{
		unsigned short *indices;

		screenQuadIndexBuffer.map((void*&)indices);
		{
			indices[0] = 0;
			indices[1] = 1;
			indices[2] = 2;

			indices[3] = 2;
			indices[4] = 1;
			indices[5] = 3;
		}
		screenQuadIndexBuffer.unmap();
	}



	sceneRenderTarget.init(rtfRGBA, CApplication::getScreenWidth() / 8, CApplication::getScreenWidth() / 8);
	sceneDepthStencilSurface.init(dssfD24X8, CApplication::getScreenWidth() / 8, CApplication::getScreenWidth() / 8);
	shadowMapRenderTarget.init(rtfRGBA, 1024, 1024);
	shadowMapDepthStencilSurface.init(dssfShadowMap, 1024, 1024);



	screenQuadVertexShader.init("data/shaders/screen_quad.vs", stVertexShader);
	#ifdef RNDR_D3D
		const char *args0[20] = { "-DRNDR_D3D" };
		screenQuadPixelShader.init("data/shaders/screen_quad.ps", stPixelShader, 1, args0);
	#else
		screenQuadPixelShader.init("data/shaders/screen_quad.ps", stPixelShader);
	#endif
	shadowMapPassVertexShader.init("data/shaders/mesh_static_shadowmap.vs", stVertexShader);
	shadowMapPassPixelShader.init("data/shaders/mesh_shadowmap.ps", stPixelShader);
	lightPassVertexShader.init("data/shaders/mesh_static_light.vs", stVertexShader);
	lightPassPixelShader.init("data/shaders/mesh_light.ps", stPixelShader);
	


	grassDiffuseMap.init();
	grassDiffuseMap.loadDataFromFile("data/textures/grass_diffuse.jpg", 0);

	brickDiffuseMap.init();
	brickDiffuseMap.loadDataFromFile("data/textures/brick_diffuse.jpg", 0);

	grassNormalMap.init();
	grassNormalMap.loadDataFromFile("data/textures/grass_normal.jpg", 0);

	brickNormalMap.init();
	brickNormalMap.loadDataFromFile("data/textures/brick_normal.jpg", 0);



	terrainMesh.init();
//	terrainMesh.importASE("data/meshes/terrain.ase");
//	terrainMesh.saveDataToFile("data/meshes/terrain.mdl");
	terrainMesh.loadDataFromFile("data/meshes/terrain.mdl");

	stuffMesh.init();
//	stuffMesh.importASE("data/meshes/stuff.ase");
//	stuffMesh.saveDataToFile("data/meshes/stuff.mdl");
	stuffMesh.loadDataFromFile("data/meshes/stuff.mdl");
}



void doRender(int deltaTime)
{
	static float delta = 0.0f; delta += 0.005f;



	terrainWorldTransform = CMatrix::rotateX(-PI/2.0f);
	stuffWorldTransform = CMatrix::rotateX(-PI/2.0f) * CMatrix::translate(0.0f, 50.0f * (sinf(delta) + 1.0f), 0.0f);

	viewProjTransform = CMatrix::lookAtRH(camera.getEye(), camera.getAt(), camera.getUp()) * CMatrix::perspectiveFovRH(PI/3.0f, (float)CApplication::getScreenWidth()/(float)CApplication::getScreenHeight(), 1.0f, 2000.0f);
	lightViewProjTransform = CMatrix::lookAtRH(-lightDirection*1000.0f, CVector3(0.0f, 0.0f, 0.0f), CVector3(0.0f, 1.0f, 0.0f)) * CMatrix::orthoRH(1200.0f, 1200.0f, 1.0f, 3000.0f);

	lightDirection = CVector3(cosf(delta), -0.5f, sinf(delta)).getNormalized();



	CRenderer::setRenderTarget(&shadowMapRenderTarget);
	CRenderer::setDepthStencilSurface(&shadowMapDepthStencilSurface);
	CRenderer::setTargetWriteState(false);
	CRenderer::setDepthWriteState(true);
	CRenderer::clear(false, true, false, CVector3(0.0f, 0.0f, 0.0f));
	{
		CRenderer::setShader(shadowMapPassVertexShader);
		{
			CShader::setVertexShaderMatrixConstant("worldTransform", terrainWorldTransform);
			CShader::setVertexShaderMatrixConstant("viewProjTransform", lightViewProjTransform);
		}
		CRenderer::setShader(shadowMapPassPixelShader);
		terrainMesh.render();

		CRenderer::setShader(shadowMapPassVertexShader);
		{
			CShader::setVertexShaderMatrixConstant("worldTransform", stuffWorldTransform);
			CShader::setVertexShaderMatrixConstant("viewProjTransform", lightViewProjTransform);
		}
		CRenderer::setShader(shadowMapPassPixelShader);
		stuffMesh.render();
	}



	CRenderer::setRenderTarget(&sceneRenderTarget);
	CRenderer::setDepthStencilSurface(&sceneDepthStencilSurface);
	CRenderer::setTargetWriteState(true);
	CRenderer::setDepthWriteState(true);
	CRenderer::clear(true, true, false, CVector3(0.5f, 0.5f, 0.5f));
	{
		CRenderer::setShader(lightPassVertexShader);
		{
			CShader::setVertexShaderMatrixConstant("worldTransform", terrainWorldTransform);
			CShader::setVertexShaderMatrixConstant("worldTransformInversed", terrainWorldTransform.getInversed());
			CShader::setVertexShaderMatrixConstant("viewProjTransform", viewProjTransform);
			CShader::setVertexShaderMatrixConstant("lightViewProjTransform", lightViewProjTransform * projTexCoordTransform);
			CShader::setVertexShaderVectorConstant("lightDirection", lightDirection, 0.0f);
		}
		CRenderer::setShader(lightPassPixelShader);
		CRenderer::setTexture(0, grassDiffuseMap);
		CRenderer::setTexture(1, grassNormalMap);
		CRenderer::setTexture(2, shadowMapDepthStencilSurface);
		terrainMesh.render();

		CRenderer::setShader(lightPassVertexShader);
		{
			CShader::setVertexShaderMatrixConstant("worldTransform", stuffWorldTransform);
			CShader::setVertexShaderMatrixConstant("worldTransformInversed", stuffWorldTransform.getInversed());
			CShader::setVertexShaderMatrixConstant("viewProjTransform", viewProjTransform);
			CShader::setVertexShaderMatrixConstant("lightViewProjTransform", lightViewProjTransform * projTexCoordTransform);
			CShader::setVertexShaderVectorConstant("lightDirection", lightDirection, 0.0f);
		}
		CRenderer::setShader(lightPassPixelShader);
		CRenderer::setTexture(0, brickDiffuseMap);
		CRenderer::setTexture(1, brickNormalMap);
		CRenderer::setTexture(2, shadowMapDepthStencilSurface);
		stuffMesh.render();
	}



	CRenderer::setRenderTarget(0);
	CRenderer::setDepthStencilSurface(0);
	CRenderer::setTargetWriteState(true);
	CRenderer::setDepthWriteState(true);
	CRenderer::clear(true, true, false, CVector3(0.5f, 0.5f, 0.5f));
	{
		CRenderer::setVertexDeclaration(screenQuadVertexDeclaration);
		CRenderer::setVertexBuffer(screenQuadVertexBuffer);
		CRenderer::setIndexBuffer(screenQuadIndexBuffer);
		CRenderer::setShader(screenQuadVertexShader);
		CRenderer::setShader(screenQuadPixelShader);
		{
			CShader::setPixelShaderFloatConstant("texelHalfWidth", 0.5f*(1.0f/(float)CApplication::getScreenWidth()));
			CShader::setPixelShaderFloatConstant("texelHalfHeight", 0.5f*(1.0f/(float)CApplication::getScreenHeight()));
		}
		CRenderer::setTexture(0, sceneRenderTarget);
		CRenderer::drawIndexedPrimitives(ptTriangleList, 4, 2);
	//	CRenderer::drawPrimitives(ptTriangleStrip, 0, 2);
	}
}



int FPS;



void showFPS()
{
	char text[100];
	sprintf(text, "FPS: %d", FPS);
	CApplication::setWindowText(text);
	FPS = 0;
}



void mouseMotionFunction()
{
	camera.horizontalAngle -= CApplication::getMouseRelX() / 1000.0f;
	camera.verticalAngle -= CApplication::getMouseRelY() / 1000.0f;
}



#ifdef WIN32
	#undef main
#endif

int main(int argc, char* argv[])
{
	#ifdef WIN32
		#if defined(DEBUG) | defined(_DEBUG)
			_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
		#endif
	#endif

	CApplication::init(0, 0, 0, false, false);
	CApplication::addTimer(1, 1000, showFPS);
	CApplication::setMouseMotionFunction(mouseMotionFunction);

	CRenderer::init();

	startPhysics();
	startRender(CApplication::getScreenWidth(), CApplication::getScreenHeight());

	while (true)
	{
		CApplication::beginLoop();
		CApplication::processEvents();

		int lastFrameTime = CApplication::getLastFrameTime();
		doPhysics(lastFrameTime);
		doRender(lastFrameTime);
		FPS++;

		CApplication::endLoop();

		if (CApplication::isKeyPressed(SDLK_ESCAPE))
		{
			terrainMesh.free();
			stuffMesh.free();

			grassDiffuseMap.free();
			brickDiffuseMap.free();

			grassNormalMap.free();
			brickNormalMap.free();

			screenQuadVertexShader.free();
			screenQuadPixelShader.free();
			shadowMapPassVertexShader.free();
			shadowMapPassPixelShader.free();
			lightPassVertexShader.free();
			lightPassPixelShader.free();

			screenQuadVertexDeclaration.free();
			screenQuadVertexBuffer.free();
			screenQuadIndexBuffer.free();

			sceneRenderTarget.free();
			sceneDepthStencilSurface.free();
			shadowMapRenderTarget.free();
			shadowMapDepthStencilSurface.free();

			CRenderer::free();
			CApplication::free();

			break;
		}
	}

	return 0;
}
