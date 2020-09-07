
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "App.h"

BaseApp *app = new App();

bool App::onMouseButton(const int x, const int y, const MouseButton button, const bool pressed)
{
	if (D3D10App::onMouseButton(x, y, button, pressed)) return true;

	if (pressed)
	{
		if (button == MOUSE_LEFT)
		{
			float cosX = cosf(wx), sinX = sinf(wx), cosY = cosf(wy), sinY = sinf(wy);
			vec3 dz(-cosX * sinY, -sinX, cosX * cosY);

			float3 pos;
			if (m_BSP.intersects(camPos, camPos + 4000.0f * dz, &pos))
			{
				float x = float(rand()) * (2 * PI / RAND_MAX);
				float y = float(rand()) * (2 * PI / RAND_MAX);
				float z = float(rand()) * (2 * PI / RAND_MAX);

				const float radius = 120.0f + float(rand()) * (80.0f / RAND_MAX);

				Decal decal;
				decal.position = pos;
				decal.radius = radius;
				if (m_RandomizeColor->isChecked())
				{
					decal.color = float3(rand() * (1.0f / RAND_MAX), rand() * (1.0f / RAND_MAX), rand() * (1.0f / RAND_MAX));
				}
				else
				{
					decal.color = float3(m_RSlider->getValue(), m_GSlider->getValue(), m_BSlider->getValue());
				}
				decal.matrix = translate(0.5f, 0.5f, 0.5f) * scale(0.5f / radius, 0.5f / radius, 0.5f / radius) * rotateZXY(x, y, z) * translate(-pos);

				m_Decals.add(decal);
			}
		}

		m_HelpDisplayTime = 0.0f;
	}

	return false;
}

void App::moveCamera(const float3 &dir)
{
	float3 newPos = camPos + dir * (speed * frameTime);

	float3 point;
	const BTri *tri;
	if (m_BSP.intersects(camPos, newPos, &point, &tri))
	{
		newPos = point + tri->plane.xyz();
	}
	m_BSP.pushSphere(newPos, 35);

	camPos = newPos;
}

void App::resetCamera()
{
	camPos = vec3(-730, 20, 2010);
	wx = 0.14f;
	wy = -2.63f;
}

void App::onSize(const int w, const int h)
{
	D3D10App::onSize(w, h);

	if (renderer)
	{
		// Make sure render targets are the size of the window
		renderer->resizeRenderTarget(m_BaseRT,   w, h, 1, 1, 1);
		renderer->resizeRenderTarget(m_NormalRT, w, h, 1, 1, 1);
		renderer->resizeRenderTarget(m_DepthRT,  w, h, 1, 1, 1);
	}
}

bool App::init()
{
	// No framework created depth buffer
	depthBits = 0;

	m_Map = new Model();
	if (!m_Map->loadObj("../Models/Corridor2/Map.obj")) return false;
	m_Map->scale(0, float3(1, 1, -1));

	uint nIndices = m_Map->getIndexCount();
	float3 *src = (float3 *) m_Map->getStream(0).vertices;
	uint *inds = m_Map->getStream(0).indices;

	for (uint i = 0; i < nIndices; i += 3)
	{
		float3 v0 = src[inds[i]];
		float3 v1 = src[inds[i + 1]];
		float3 v2 = src[inds[i + 2]];

		m_BSP.addTriangle(v0, v1, v2);
	}
	m_BSP.build();

	m_Map->computeTangentSpace(true);

	m_Sphere = new Model();
	m_Sphere->createSphere(3);

	// Initialize all lights
	m_Lights[ 0].position = float3( 576, 96,    0); m_Lights[ 0].radius = 640.0f;
	m_Lights[ 1].position = float3( 0,   96,  576); m_Lights[ 1].radius = 640.0f;
	m_Lights[ 2].position = float3(-576, 96,    0); m_Lights[ 2].radius = 640.0f;
	m_Lights[ 3].position = float3( 0,   96, -576); m_Lights[ 3].radius = 640.0f;
	m_Lights[ 4].position = float3(1792, 96,  320); m_Lights[ 4].radius = 550.0f;
	m_Lights[ 5].position = float3(1792, 96, -320); m_Lights[ 5].radius = 550.0f;
	m_Lights[ 6].position = float3(-192, 96, 1792); m_Lights[ 6].radius = 550.0f;
	m_Lights[ 7].position = float3(-832, 96, 1792); m_Lights[ 7].radius = 550.0f;
	m_Lights[ 8].position = float3(1280, 32,  192); m_Lights[ 8].radius = 450.0f;
	m_Lights[ 9].position = float3(1280, 32, -192); m_Lights[ 9].radius = 450.0f;
	m_Lights[10].position = float3(-320, 32, 1280); m_Lights[10].radius = 450.0f;
	m_Lights[11].position = float3(-704, 32, 1280); m_Lights[11].radius = 450.0f;
	m_Lights[12].position = float3( 960, 32,  640); m_Lights[12].radius = 450.0f;
	m_Lights[13].position = float3( 960, 32, -640); m_Lights[13].radius = 450.0f;
	m_Lights[14].position = float3( 640, 32, -960); m_Lights[14].radius = 450.0f;
	m_Lights[15].position = float3(-640, 32, -960); m_Lights[15].radius = 450.0f;
	m_Lights[16].position = float3(-960, 32,  640); m_Lights[16].radius = 450.0f;
	m_Lights[17].position = float3(-960, 32, -640); m_Lights[17].radius = 450.0f;
	m_Lights[18].position = float3( 640, 32,  960); m_Lights[18].radius = 450.0f;

	// Init GUI components
	int tab = configDialog->addTab("Decals");
	configDialog->addWidget(tab, new Label(0, 0, 192, 36, "Decal color"));
	configDialog->addWidget(tab, new Label(0, 35, 70, 36, "Red"));
	configDialog->addWidget(tab, new Label(0, 65, 70, 36, "Green"));
	configDialog->addWidget(tab, new Label(0, 95, 70, 36, "Blue"));
	configDialog->addWidget(tab, m_RSlider = new Slider(70, 40,  200, 24, 0.0f, 1.0f, 0.0f));
	configDialog->addWidget(tab, m_GSlider = new Slider(70, 70,  200, 24, 0.0f, 1.0f, 0.0f));
	configDialog->addWidget(tab, m_BSlider = new Slider(70, 100, 200, 24, 0.0f, 1.0f, 1.0f));
	configDialog->addWidget(tab, m_RandomizeColor = new CheckBox(0, 130, 200, 36, "Randomize", false));

	m_HelpDisplayTime = 6.0f;

	return true;
}

void App::exit()
{
	delete m_Sphere;
	delete m_Map;
}

bool App::initAPI()
{
	// Override the user's MSAA settings
	return D3D10App::initAPI(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN, 1, NO_SETTING_CHANGE);
}

void App::exitAPI()
{
	D3D10App::exitAPI();
}

bool App::load()
{
	// Shaders
	if ((m_FillBuffers = renderer->addShader("FillBuffers.shd")) == SHADER_NONE) return false;
	if ((m_Ambient     = renderer->addShader("Ambient.shd")) == SHADER_NONE) return false;
	if ((m_Lighting    = renderer->addShader("Lighting.shd")) == SHADER_NONE) return false;
	if ((m_Decal       = renderer->addShader("Decals.shd")) == SHADER_NONE) return false;

	// Samplerstates
	if ((m_BaseFilter = renderer->addSamplerState(TRILINEAR_ANISO, WRAP, WRAP, WRAP)) == SS_NONE) return false;
	if ((m_PointClamp = renderer->addSamplerState(NEAREST, CLAMP, CLAMP, CLAMP)) == SS_NONE) return false;
	if ((m_DecalSS    = renderer->addSamplerState(LINEAR, BORDER, BORDER, BORDER, 0, 1, 0, float4(0, 0, 0, 0))) == SS_NONE) return false;

	// Main render targets
	if ((m_BaseRT   = renderer->addRenderTarget(width, height, 1, 1, 1, FORMAT_RGBA8,  1, SS_NONE)) == TEXTURE_NONE) return false;
	if ((m_NormalRT = renderer->addRenderTarget(width, height, 1, 1, 1, FORMAT_RGBA8S, 1, SS_NONE)) == TEXTURE_NONE) return false;
	if ((m_DepthRT  = renderer->addRenderDepth (width, height, 1,       FORMAT_D16,    1, SS_NONE, SAMPLE_DEPTH)) == TEXTURE_NONE) return false;

	// Textures
	if ((m_BaseTex[0] = renderer->addTexture  ("../Textures/wood.dds",                    true, m_BaseFilter)) == TEXTURE_NONE) return false;
	if ((m_BumpTex[0] = renderer->addNormalMap("../Textures/woodBump.dds", FORMAT_RGBA8S, true, m_BaseFilter)) == TEXTURE_NONE) return false;

	if ((m_BaseTex[1] = renderer->addTexture  ("../Textures/Tx_imp_wall_01_small.dds",              true, m_BaseFilter)) == TEXTURE_NONE) return false;
	if ((m_BumpTex[1] = renderer->addNormalMap("../Textures/Tx_imp_wall_01Bump.dds", FORMAT_RGBA8S, true, m_BaseFilter)) == TEXTURE_NONE) return false;

	if ((m_BaseTex[2] = renderer->addTexture  ("../Textures/floor_wood_4.dds",                    true, m_BaseFilter)) == TEXTURE_NONE) return false;
	if ((m_BumpTex[2] = renderer->addNormalMap("../Textures/floor_wood_4Bump.dds", FORMAT_RGBA8S, true, m_BaseFilter)) == TEXTURE_NONE) return false;

	if ((m_BaseTex[3] = renderer->addTexture  ("../Textures/floor_wood_3.dds",                    true, m_BaseFilter)) == TEXTURE_NONE) return false;
	if ((m_BumpTex[3] = renderer->addNormalMap("../Textures/floor_wood_3Bump.dds", FORMAT_RGBA8S, true, m_BaseFilter)) == TEXTURE_NONE) return false;

	if ((m_BaseTex[4] = renderer->addTexture  ("../Textures/light2.dds",                    true, m_BaseFilter)) == TEXTURE_NONE) return false;
	if ((m_BumpTex[4] = renderer->addNormalMap("../Textures/light2Bump.dds", FORMAT_RGBA8S, true, m_BaseFilter)) == TEXTURE_NONE) return false;



	// Generate a decal texture with some noise
	initNoise();

	const int size = 128;

	Image img;
	ubyte *dest = img.create(FORMAT_R8, size, size, size, 1);

	float3 pos;
	for (int z = 0; z < size; z++)
	{
		pos.z = z * (2.0f / (size - 1)) - 1.0f;
		for (int y = 0; y < size; y++)
		{
			pos.y = y * (2.0f / (size - 1)) - 1.0f;
			for (int x = 0; x < size; x++)
			{
				pos.x = x * (2.0f / (size - 1)) - 1.0f;

				float d = 1.0f - dot(pos, pos);

				float3 p = pos * d * (1.5f * float3(3.7192f, 5.2398f, 4.38194f));
				d -= (noise3(p.x, p.y, p.z) * 0.5f + 0.5f);
				d = saturate(d * 1.3f);

				*dest++ = ubyte(d * 255.0f + 0.5f);
			}
		}
	}
	if ((m_DecalTex = renderer->addTexture(img, false, m_DecalSS)) == TEXTURE_NONE) return false;


	// Blendstates
	if ((m_BlendAdd = renderer->addBlendState(ONE, ONE)) == BS_NONE) return false;
	if ((m_BlendDecal = renderer->addBlendState(SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ZERO, ONE_MINUS_SRC_ALPHA, BM_ADD, BM_ADD)) == BS_NONE) return false;

	// Depth states - use reversed depth (1 to 0) to improve precision
	if ((m_DepthTest = renderer->addDepthState(true, true, GEQUAL)) == DS_NONE) return false;

	// Upload map to vertex/index buffer
	if (!m_Map->makeDrawable(renderer, true, m_FillBuffers)) return false;
	if (!m_Sphere->makeDrawable(renderer, true, m_Lighting)) return false;

	return true;
}

void App::unload(){

}

void App::drawFrame(){
	const float near_plane = 20.0f;
	const float far_plane = 4000.0f;

	// Reversed depth
	float4x4 projection = toD3DProjection(perspectiveMatrixY(1.2f, width, height, far_plane, near_plane));
	float4x4 view = rotateXY(-wx, -wy);
	view.translate(-camPos);
	float4x4 viewProj = projection * view;
	// Pre-scale-bias the matrix so we can use the screen position directly
	float4x4 viewProjInv = (!viewProj) * (translate(-1.0f, 1.0f, 0.0f) * scale(2.0f, -2.0f, 1.0f));


	TextureID bufferRTs[] = { m_BaseRT, m_NormalRT };
	renderer->changeRenderTargets(bufferRTs, elementsOf(bufferRTs), m_DepthRT);

		renderer->clear(false, true, false, NULL, 0.0f);

		/*
			Main scene pass.
			This is where the buffers are filled for the later deferred passes.
		*/
		renderer->reset();
		renderer->setRasterizerState(cullBack);
		renderer->setShader(m_FillBuffers);
		renderer->setShaderConstant4x4f("ViewProj", viewProj);
		renderer->setSamplerState("Filter", m_BaseFilter);
		renderer->setDepthState(m_DepthTest);
		renderer->apply();

		const uint batch_count = m_Map->getBatchCount();
		for (uint i = 0; i < batch_count; i++)
		{
			renderer->setTexture("Base", m_BaseTex[i]);
			renderer->setTexture("Bump", m_BumpTex[i]);
			renderer->applyTextures();

			m_Map->drawBatch(renderer, i);
		}

	renderer->changeRenderTarget(m_BaseRT);

		/*
			Decals pass.
			Only updating the diffuse part of the g-buffer.
		*/
		renderer->reset();
		renderer->setRasterizerState(cullFront);
		renderer->setShader(m_Decal);
		renderer->setShaderConstant4x4f("ViewProj", viewProj);
		renderer->setShaderConstant2f("PixelSize", float2(1.0f / width, 1.0f / height));
		renderer->setTexture("Depth", m_DepthRT);
		renderer->setTexture("Decal", m_DecalTex);
		renderer->setSamplerState("DepthFilter", m_PointClamp);
		renderer->setSamplerState("DecalFilter", m_DecalSS);
		renderer->setDepthState(noDepthTest);
		renderer->setBlendState(m_BlendDecal);
		renderer->apply();


		const uint decal_count = m_Decals.getCount();
		for (uint i = 0; i < decal_count; i++)
		{
			renderer->setShaderConstant3f("Pos", m_Decals[i].position);
			renderer->setShaderConstant1f("Radius", m_Decals[i].radius);
			renderer->setShaderConstant3f("Color", m_Decals[i].color);
			renderer->setShaderConstant4x4f("ScreenToLocal", m_Decals[i].matrix * viewProjInv);
			renderer->applyConstants();

			m_Sphere->draw(renderer);
		}

	renderer->changeToMainFramebuffer();

	/*
		Deferred ambient pass.
	*/
	renderer->reset();
	renderer->setRasterizerState(cullNone);
	renderer->setShader(m_Ambient);
	renderer->setTexture("Base", m_BaseRT);
	renderer->setSamplerState("Filter", m_PointClamp);
	renderer->setDepthState(noDepthTest);
	renderer->apply();


	device->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	device->Draw(3, 0);


	/*
		Deferred lighting pass.
	*/
	renderer->reset();

	renderer->setDepthState(noDepthTest);
	renderer->setShader(m_Lighting);

	renderer->setRasterizerState(cullFront);
	renderer->setBlendState(m_BlendAdd);
	renderer->setShaderConstant4x4f("ViewProj", viewProj);
	renderer->setShaderConstant4x4f("ViewProjInv", viewProjInv * scale(1.0f / width, 1.0f / height, 1.0f));
	renderer->setShaderConstant3f("CamPos", camPos);
	renderer->setTexture("Base", m_BaseRT);
	renderer->setTexture("Normal", m_NormalRT);
	renderer->setTexture("Depth", m_DepthRT);
	renderer->apply();

	float2 zw = projection.rows[2].zw();
	for (uint i = 0; i < LIGHT_COUNT; i++)
	{
		float3 lightPos = m_Lights[i].position;
		float radius = m_Lights[i].radius;
		float invRadius = 1.0f / radius;

		// Compute z-bounds
		float4 lPos = view * float4(lightPos, 1.0f);
		float z1 = lPos.z + radius;

		if (z1 > near_plane)
		{
			float z0 = max(lPos.z - radius, near_plane);

			float2 zBounds;
			zBounds.y = saturate(zw.x + zw.y / z0);
			zBounds.x = saturate(zw.x + zw.y / z1);

			renderer->setShaderConstant3f("LightPos", lightPos);
			renderer->setShaderConstant1f("Radius", radius);
			renderer->setShaderConstant1f("InvRadius", invRadius);
			renderer->setShaderConstant2f("ZBounds", zBounds);
			renderer->applyConstants();

			m_Sphere->draw(renderer);
		}
	}


	// Display help text
	if (m_HelpDisplayTime > 0)
	{
		if (configDialog->isVisible())
		{
			m_HelpDisplayTime = 0;
		}
		else
		{
			m_HelpDisplayTime -= min(frameTime, 0.1f);

			renderer->drawText("Press mouse button to place a\ndecal at the wall in front of you.\n\nChange decal color on the F1 dialog.",
				width * 0.5f - 220, height * 0.5f - 75, 30, 38, defaultFont, linearClamp, blendSrcAlpha, noDepthTest);
		}
	}
}
