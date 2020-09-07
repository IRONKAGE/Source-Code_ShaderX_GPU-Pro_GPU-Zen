#include "stdafx.h"
#include "app.h"
#include "dds.h"

#define SAFE_RELEASE( x ) { if ( x ) { x->Release(); x = nullptr; } }

CApp gApp;

char const* ImagePathArr[] = { "atrium.dds", "backyard.dds", "desk.dds", "memorial.dds", "yucca.dds" };
const uint32_t BC_BLOCK_SIZE = 4;

struct SShaderCB
{
	Vec2 m_screenSizeRcp;
	unsigned m_textureSizeInBlocks[2];

	Vec2 m_imageSizeRcp;
	Vec2 m_texelBias;

	float m_texelScale;
	float m_exposure;
	uint32_t m_blitMode;
	uint32_t m_padding;
};

// https://gist.github.com/rygorous/2144712
static float HalfToFloat(uint16_t h)
{
	union FP32
	{
		uint32_t    u;
		float       f;
		struct
		{
			unsigned Mantissa : 23;
			unsigned Exponent : 8;
			unsigned Sign : 1;
		};
	};

	static const FP32 magic = { (254 - 15) << 23 };
	static const FP32 was_infnan = { (127 + 16) << 23 };

	FP32 o;
	o.u = (h & 0x7fff) << 13;     // exponent/mantissa bits
	o.f *= magic.f;                 // exponent adjust
	if (o.f >= was_infnan.f)        // make sure Inf/NaN survive
		o.u |= 255 << 23;
	o.u |= (h & 0x8000) << 16;    // sign bit
	return o.f;
}

uint32_t DivideAndRoundUp(uint32_t x, uint32_t divisor)
{
	return (x + divisor - 1) / divisor;
}

CApp::CApp()
{
}

CApp::~CApp()
{
	DestoryImage();
	DestroyTargets();
	DestroyShaders();
	SAFE_RELEASE(m_ctx);
	SAFE_RELEASE(m_swapChain);
	SAFE_RELEASE(m_device);
}

bool CApp::Init(HWND windowHandle)
{
	m_windowHandle = windowHandle;

	RECT clientRect;
	GetClientRect(windowHandle, &clientRect);
	m_backbufferWidth = clientRect.right - clientRect.left;
	m_backbufferHeight = clientRect.bottom - clientRect.top;

	D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
	D3D_FEATURE_LEVEL retFeatureLevel;

	DXGI_SWAP_CHAIN_DESC swapDesc;
	ZeroMemory(&swapDesc, sizeof(swapDesc));
	swapDesc.BufferDesc.Width = m_backbufferWidth;
	swapDesc.BufferDesc.Height = m_backbufferHeight;
	swapDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	swapDesc.BufferDesc.RefreshRate.Numerator = 60;
	swapDesc.BufferDesc.RefreshRate.Denominator = 1;
	swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapDesc.BufferCount = 2;
	swapDesc.OutputWindow = windowHandle;
	swapDesc.SampleDesc.Count = 1;
	swapDesc.SampleDesc.Quality = 0;
	swapDesc.Windowed = true;

	unsigned flags = 0;
#ifdef _DEBUG
	flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

	HRESULT res;
	res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, 0, flags, featureLevels, ARRAYSIZE(featureLevels), D3D11_SDK_VERSION, &swapDesc, &m_swapChain, &m_device, &retFeatureLevel, &m_ctx);
	_ASSERT(SUCCEEDED(res));

	ID3D11Texture2D* backBuffer = NULL;
	res = m_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&backBuffer);
	_ASSERT(SUCCEEDED(res));

	res = m_device->CreateRenderTargetView(backBuffer, nullptr, &m_backBufferView);
	_ASSERT(SUCCEEDED(res));
	backBuffer->Release();

	CreateImage();
	CreateShaders();
	CreateTargets();
	CreateQueries();
	CreateConstantBuffer();

	HRESULT hr;
	D3D11_SAMPLER_DESC samplerDesc =
	{
		D3D11_FILTER_MIN_MAG_MIP_POINT,
		D3D11_TEXTURE_ADDRESS_BORDER,
		D3D11_TEXTURE_ADDRESS_BORDER,
		D3D11_TEXTURE_ADDRESS_BORDER,
		0.0f,
		1,
		D3D11_COMPARISON_ALWAYS,
		0.0f,
		0.0f,
		0.0f,
		0.0f,
		0.0f,
		D3D11_FLOAT32_MAX
	};
	hr = m_device->CreateSamplerState(&samplerDesc, &m_pointSampler);
	_ASSERT(SUCCEEDED(hr));

	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd, sizeof(bd));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(uint16_t) * 4;
	bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bd.CPUAccessFlags = 0;

	uint16_t indices[] = { 0, 1, 2, 3 };
	D3D11_SUBRESOURCE_DATA initData;
	ZeroMemory(&initData, sizeof(initData));
	initData.pSysMem = indices;

	hr = m_device->CreateBuffer(&bd, &initData, &m_ib);
	_ASSERT(SUCCEEDED(hr));

	return true;
}

void CApp::CreateTargets()
{
	D3D11_TEXTURE2D_DESC texDesc;
	texDesc.Width = DivideAndRoundUp(m_imageWidth, BC_BLOCK_SIZE);
	texDesc.Height = DivideAndRoundUp(m_imageHeight, BC_BLOCK_SIZE);
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Usage = D3D11_USAGE_DEFAULT;
	texDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	texDesc.CPUAccessFlags = 0;
	texDesc.MiscFlags = 0;
	HRESULT hr = m_device->CreateTexture2D(&texDesc, nullptr, &m_compressTargetRes);
	_ASSERT(SUCCEEDED(hr));

	hr = m_device->CreateUnorderedAccessView(m_compressTargetRes, nullptr, &m_compressTargetUAV);
	_ASSERT(SUCCEEDED(hr));

	texDesc.Width = m_imageWidth;
	texDesc.Height = m_imageHeight;
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Usage = D3D11_USAGE_DEFAULT;
	texDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	texDesc.CPUAccessFlags = 0;
	texDesc.MiscFlags = 0;
	hr = m_device->CreateTexture2D(&texDesc, nullptr, &m_tmpTargetRes);
	_ASSERT(SUCCEEDED(hr));

	hr = m_device->CreateRenderTargetView(m_tmpTargetRes, nullptr, &m_tmpTargetView);
	_ASSERT(SUCCEEDED(hr));

	texDesc.Width = m_imageWidth;
	texDesc.Height = m_imageHeight;
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Usage = D3D11_USAGE_STAGING;
	texDesc.BindFlags = 0;
	texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	texDesc.MiscFlags = 0;
	hr = m_device->CreateTexture2D(&texDesc, nullptr, &m_tmpStagingRes);
	_ASSERT(SUCCEEDED(hr));
}

void CApp::DestroyTargets()
{
	SAFE_RELEASE(m_compressTargetUAV);
	SAFE_RELEASE(m_compressTargetRes);
	SAFE_RELEASE(m_tmpTargetView);
	SAFE_RELEASE(m_tmpTargetRes);
	SAFE_RELEASE(m_tmpStagingRes);
}

void CApp::CreateQueries()
{
	D3D11_QUERY_DESC queryDesc;
	queryDesc.MiscFlags = 0;

	for (unsigned i = 0; i < MAX_QUERY_FRAME_NUM; ++i)
	{
		queryDesc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
		m_device->CreateQuery(&queryDesc, &m_disjointQueries[i]);

		queryDesc.Query = D3D11_QUERY_TIMESTAMP;
		m_device->CreateQuery(&queryDesc, &m_timeBeginQueries[i]);
		m_device->CreateQuery(&queryDesc, &m_timeEndQueries[i]);
	}
}

void CApp::CreateConstantBuffer()
{
	D3D11_BUFFER_DESC desc;
	ZeroMemory(&desc, sizeof(desc));
	desc.Usage = D3D11_USAGE_DYNAMIC;
	desc.ByteWidth = sizeof(SShaderCB);
	desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	m_device->CreateBuffer(&desc, nullptr, &m_constantBuffer);
}

void CApp::CreateImage()
{
	SImage img;
	DDS::LoadA16B16G16R16F(ImagePathArr[m_imageID], img);

	m_imageWidth = img.m_width;
	m_imageHeight = img.m_height;

	D3D11_SUBRESOURCE_DATA initialData;
	initialData.pSysMem = img.m_data;
	initialData.SysMemPitch = img.m_width * 4 * 2;
	initialData.SysMemSlicePitch = 0;

	D3D11_TEXTURE2D_DESC desc;
	ZeroMemory(&desc, sizeof(desc));
	desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
	desc.Width = img.m_width;
	desc.Height = img.m_height;
	desc.MipLevels = 1;
	desc.ArraySize = 1;
	desc.Usage = D3D11_USAGE_IMMUTABLE;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	HRESULT hr = m_device->CreateTexture2D(&desc, &initialData, &m_sourceTextureRes);
	_ASSERT(SUCCEEDED(hr));

	D3D11_SHADER_RESOURCE_VIEW_DESC resViewDesc;
	resViewDesc.Format = desc.Format;
	resViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	resViewDesc.Texture2D.MostDetailedMip = 0;
	resViewDesc.Texture2D.MipLevels = desc.MipLevels;
	hr = m_device->CreateShaderResourceView(m_sourceTextureRes, &resViewDesc, &m_sourceTextureView);
	_ASSERT(SUCCEEDED(hr));

	desc.Format = DXGI_FORMAT_BC6H_UF16;
	desc.Usage = D3D11_USAGE_DEFAULT;
	hr = m_device->CreateTexture2D(&desc, nullptr, &m_compressedTextureRes);
	_ASSERT(SUCCEEDED(hr));

	resViewDesc.Format = desc.Format;
	resViewDesc.Texture2D.MostDetailedMip = 0;
	resViewDesc.Texture2D.MipLevels = desc.MipLevels;

	hr = m_device->CreateShaderResourceView(m_compressedTextureRes, &resViewDesc, &m_compressedTextureView);
	_ASSERT(SUCCEEDED(hr));
}

void CApp::DestoryImage()
{
	SAFE_RELEASE(m_compressedTextureView);
	SAFE_RELEASE(m_compressedTextureRes);
	SAFE_RELEASE(m_sourceTextureView);
	SAFE_RELEASE(m_sourceTextureRes);
}

void CApp::CreateShaders()
{
	unsigned shaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
	shaderFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_PREFER_FLOW_CONTROL;
#endif

	HRESULT hr;
	ID3DBlob* shaderBlob = nullptr;
	ID3DBlob* errorBlob = nullptr;

	// Compression compute shaders
	for (uint32_t ModeIndex = 0; ModeIndex < COMPRESSION_MODE_NUM; ++ModeIndex)
	{
		D3D_SHADER_MACRO macros[2];
		macros[0].Name = "QUALITY";
		macros[0].Definition = (ModeIndex == 0 ? "0" : "1");
		macros[1].Name = nullptr;
		macros[1].Definition = nullptr;

		hr = D3DCompileFromFile(L"compress.hlsl", macros, nullptr, "CSMain", "cs_5_0", shaderFlags, 0, &shaderBlob, &errorBlob);
		if (SUCCEEDED(hr))
		{
			m_device->CreateComputeShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), nullptr, &m_compressCS[ModeIndex]);
		}
		else
		{
			OutputDebugStringA((char const*)errorBlob->GetBufferPointer());
		}
	}

	// Blit vertex and pixel shader
	{
		hr = D3DCompileFromFile(L"blit.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", shaderFlags, 0, &shaderBlob, &errorBlob);
		if (SUCCEEDED(hr))
		{
			m_device->CreateVertexShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), nullptr, &m_blitVS);
		}
		else
		{
			OutputDebugStringA((char const*)errorBlob->GetBufferPointer());
		}

		hr = D3DCompileFromFile(L"blit.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", shaderFlags, 0, &shaderBlob, &errorBlob);
		if (SUCCEEDED(hr))
		{
			m_device->CreatePixelShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), nullptr, &m_blitPS);
		}
		else
		{
			OutputDebugStringA((char const*)errorBlob->GetBufferPointer());
		}
	}
}

void CApp::DestroyShaders()
{
	SAFE_RELEASE(m_blitVS);
	SAFE_RELEASE(m_blitPS);

	for (unsigned i = 0; i < ARRAYSIZE(m_compressCS); ++i)
	{
		SAFE_RELEASE(m_compressCS[i]);
	}
}

void CApp::Release()
{
	DestroyShaders();
}

void CApp::OnKeyDown(WPARAM wParam)
{
	switch (wParam)
	{
	case 'R':
		DestroyShaders();
		CreateShaders();
		m_updateRMSE = true;
		OutputDebugStringA("Recompiled shaders\n");
		break;

	case 'N':
		m_imageID = (m_imageID + 1) % ARRAYSIZE(ImagePathArr);
		DestoryImage();
		DestroyTargets();
		CreateImage();
		CreateTargets();
		m_imageZoom = 0.0f;
		m_texelScale = 1.0f;
		m_texelBias.x = 0.0f;
		m_texelBias.y = 0.0f;
		m_imageExposure = 0.0f;
		m_updateRMSE = true;
		break;

	case 'E':
		// Flip between source and compressed image
		m_blitMode = (m_blitMode + 1) % 2;
		m_updateTitle = true;
		break;

	case '1':
		m_blitMode = 0;
		m_updateTitle = true;
		break;

	case '2':
		m_blitMode = 1;
		m_updateTitle = true;
		break;

	case '3':
		m_blitMode = 2;
		m_updateTitle = true;
		break;

	case '4':
		m_blitMode = 3;
		m_updateTitle = true;
		break;

	case 'Q':
		m_compressionMode = (m_compressionMode + 1) % COMPRESSION_MODE_NUM;
		m_updateTitle = true;
		m_updateRMSE = true;
		break;

	case VK_ADD:
		m_imageExposure += 0.1f;
		m_updateTitle = true;
		break;

	case VK_SUBTRACT:
		m_imageExposure -= 0.1f;
		m_updateTitle = true;
		break;
	}
}

void CApp::OnLButtonDown(int mouseX, int mouseY)
{
	m_dragEnabled = true;
	m_dragStart.x = m_texelBias.x + mouseX * m_texelScale;
	m_dragStart.y = m_texelBias.y + mouseY * m_texelScale;
}

void CApp::OnLButtonUp(int mouseX, int mouseY)
{
	m_dragEnabled = false;
}

void CApp::OnMouseMove(int mouseX, int mouseY)
{
	if (m_dragEnabled && GetKeyState(VK_LBUTTON) >= 0)
	{
		m_dragEnabled = false;
	}

	if (m_dragEnabled)
	{
		m_texelBias.x = m_dragStart.x - mouseX * m_texelScale;
		m_texelBias.y = m_dragStart.y - mouseY * m_texelScale;
	}
}

void CApp::OnMouseWheel(int zDelta)
{
	m_imageZoom -= zDelta * 0.001f;
	m_texelScale = powf(2.0f, m_imageZoom);
}

void CApp::OnResize()
{
	RECT clientRect;
	GetClientRect(m_windowHandle, &clientRect);
	unsigned const newBackbufferWidth = max(clientRect.right - clientRect.left, 64);
	unsigned const newBackbufferHeight = max(clientRect.bottom - clientRect.top, 64);

	if (m_backbufferWidth != newBackbufferWidth && m_backbufferHeight != newBackbufferHeight)
	{
		m_ctx->ClearState();
		SAFE_RELEASE(m_backBufferView);
		m_swapChain->ResizeBuffers(2, newBackbufferWidth, newBackbufferHeight, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 0);

		ID3D11Texture2D* backBuffer = nullptr;
		HRESULT hr = m_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&backBuffer);
		_ASSERT(SUCCEEDED(hr));

		hr = m_device->CreateRenderTargetView(backBuffer, nullptr, &m_backBufferView);
		_ASSERT(SUCCEEDED(hr));
		backBuffer->Release();

		m_backbufferWidth = newBackbufferWidth;
		m_backbufferHeight = newBackbufferHeight;
	}
}

void CApp::UpdateTitle()
{
	const wchar_t* blitModeNames[BLIT_MODE_NUM] =
	{
		L"Source",
		L"Compressed",
		L"DiffRGB",
		L"DiffLum"
	};

	wchar_t title[256];
	title[0] = 0;
	swprintf(title, ARRAYSIZE(title), L"Time:%.3fms rgbRMSLE:%.4f lumRMSLE:%.4f [q]Mode:%s [1,2,3,4]Show:%s [-/+]Exposure:%.1f [n]%S%dx%d [r]Reloadshaders",
		m_compressionTime, m_rgbRMSLE, m_lumRMSLE, m_compressionMode == 1 ? L"Quality" : L"Fast", blitModeNames[m_blitMode], m_imageExposure, ImagePathArr[m_imageID], m_imageWidth, m_imageHeight);

	SetWindowText(m_windowHandle, title);
}

void CApp::Render()
{
	m_ctx->ClearState();

	m_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	m_ctx->IASetIndexBuffer(m_ib, DXGI_FORMAT_R16_UINT, 0);

	SShaderCB shaderCB;
	shaderCB.m_textureSizeInBlocks[0] = DivideAndRoundUp(m_imageWidth, BC_BLOCK_SIZE);
	shaderCB.m_textureSizeInBlocks[1] = DivideAndRoundUp(m_imageHeight, BC_BLOCK_SIZE);
	shaderCB.m_imageSizeRcp.x = 1.0f / m_imageWidth;
	shaderCB.m_imageSizeRcp.y = 1.0f / m_imageHeight;
	shaderCB.m_screenSizeRcp.x = 1.0f / m_backbufferWidth;
	shaderCB.m_screenSizeRcp.y = 1.0f / m_backbufferHeight;
	shaderCB.m_texelBias = m_texelBias;
	shaderCB.m_texelScale = m_texelScale;
	shaderCB.m_exposure = exp(m_imageExposure);
	shaderCB.m_blitMode = m_blitMode;

	D3D11_MAPPED_SUBRESOURCE mappedRes;
	m_ctx->Map(m_constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedRes);
	memcpy(mappedRes.pData, &shaderCB, sizeof(shaderCB));
	m_ctx->Unmap(m_constantBuffer, 0);

	m_ctx->Begin(m_disjointQueries[m_frameID % MAX_QUERY_FRAME_NUM]);
	m_ctx->End(m_timeBeginQueries[m_frameID % MAX_QUERY_FRAME_NUM]);

	if (m_compressCS[m_compressionMode])
	{
		m_ctx->CSSetShader(m_compressCS[m_compressionMode], nullptr, 0);
		m_ctx->CSSetUnorderedAccessViews(0, 1, &m_compressTargetUAV, nullptr);
		m_ctx->CSSetShaderResources(0, 1, &m_sourceTextureView);
		m_ctx->CSSetSamplers(0, 1, &m_pointSampler);
		m_ctx->CSSetConstantBuffers(0, 1, &m_constantBuffer);

		uint32_t threadsX = 8;
		uint32_t threadsY = 8;
		m_ctx->Dispatch(DivideAndRoundUp(m_imageWidth, BC_BLOCK_SIZE * threadsX), DivideAndRoundUp(m_imageHeight, BC_BLOCK_SIZE * threadsY), 1);
	}

	m_ctx->End(m_timeEndQueries[m_frameID % MAX_QUERY_FRAME_NUM]);
	m_ctx->End(m_disjointQueries[m_frameID % MAX_QUERY_FRAME_NUM]);

	m_ctx->CopyResource(m_compressedTextureRes, m_compressTargetRes);

	if (m_blitVS && m_blitPS)
	{
		m_ctx->OMSetRenderTargets(1, &m_backBufferView, nullptr);
		D3D11_VIEWPORT vp;
		vp.Width = (float)m_backbufferWidth;
		vp.Height = (float)m_backbufferHeight;
		vp.MinDepth = 0.0f;
		vp.MaxDepth = 1.0f;
		vp.TopLeftX = 0;
		vp.TopLeftY = 0;
		m_ctx->RSSetViewports(1, &vp);

		m_ctx->VSSetShader(m_blitVS, nullptr, 0);
		m_ctx->PSSetShader(m_blitPS, nullptr, 0);
		m_ctx->PSSetShaderResources(0, 1, &m_sourceTextureView);
		m_ctx->PSSetShaderResources(1, 1, &m_compressedTextureView);
		m_ctx->PSSetSamplers(0, 1, &m_pointSampler);
		m_ctx->PSSetConstantBuffers(0, 1, &m_constantBuffer);

		m_ctx->DrawIndexed(4, 0, 0);
	}

	if (m_updateRMSE)
	{
		UpdateRMSE();
		m_updateRMSE = false;
	}

	++m_frameID;
	m_swapChain->Present(0, 0);

	D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjointData;
	uint64_t timeStart;
	uint64_t timeEnd;

	if (m_frameID > m_frameID % MAX_QUERY_FRAME_NUM)
	{
		while (m_ctx->GetData(m_disjointQueries[m_frameID % MAX_QUERY_FRAME_NUM], &disjointData, sizeof(disjointData), 0) != S_OK)
		{
			int e = 0;
		}

		while (m_ctx->GetData(m_timeBeginQueries[m_frameID % MAX_QUERY_FRAME_NUM], &timeStart, sizeof(timeStart), 0) != S_OK)
		{
			int e = 0;
		}

		while (m_ctx->GetData(m_timeEndQueries[m_frameID % MAX_QUERY_FRAME_NUM], &timeEnd, sizeof(timeEnd), 0) != S_OK)
		{
			int e = 0;
		}

		if (!disjointData.Disjoint)
		{
			uint64_t delta = (timeEnd - timeStart) * 1000;
			m_timeAcc += delta / (float)disjointData.Frequency;
			++m_timeAccSampleNum;
		}

		if (m_timeAccSampleNum > 100)
		{
			m_compressionTime = m_timeAcc / m_timeAccSampleNum;
			m_timeAcc = 0.0f;
			m_timeAccSampleNum = 0;
			m_updateTitle = true;
		}
	}

	if (m_updateTitle)
	{
		UpdateTitle();
		m_updateTitle = false;
	}
}

void CApp::CopyTexture(Vec3* image, ID3D11ShaderResourceView* srcView)
{
	if (m_blitVS && m_blitPS)
	{
		m_ctx->OMSetRenderTargets(1, &m_tmpTargetView, nullptr);
		D3D11_VIEWPORT vp;
		vp.Width = (float)m_imageWidth;
		vp.Height = (float)m_imageHeight;
		vp.MinDepth = 0.0f;
		vp.MaxDepth = 1.0f;
		vp.TopLeftX = 0;
		vp.TopLeftY = 0;
		m_ctx->RSSetViewports(1, &vp);

		m_ctx->VSSetShader(m_blitVS, nullptr, 0);
		m_ctx->PSSetShader(m_blitPS, nullptr, 0);
		m_ctx->PSSetShaderResources(0, 1, &srcView);
		m_ctx->PSSetShaderResources(1, 1, &srcView);
		m_ctx->PSSetSamplers(0, 1, &m_pointSampler);

		m_ctx->DrawIndexed(4, 0, 0);
		m_ctx->CopyResource(m_tmpStagingRes, m_tmpTargetRes);

		D3D11_MAPPED_SUBRESOURCE mappedRes;
		m_ctx->Map(m_tmpStagingRes, 0, D3D11_MAP_READ, 0, &mappedRes);
		if (mappedRes.pData)
		{
			for (unsigned y = 0; y < m_imageHeight; ++y)
			{
				for (unsigned x = 0; x < m_imageWidth; ++x)
				{
					uint16_t tmp[4];
					memcpy(&tmp, (uint8_t*)mappedRes.pData + mappedRes.RowPitch * y + x * sizeof(tmp), sizeof(tmp));

					image[x + y * m_imageWidth].x = HalfToFloat(tmp[0]);
					image[x + y * m_imageWidth].y = HalfToFloat(tmp[1]);
					image[x + y * m_imageWidth].z = HalfToFloat(tmp[2]);
				}
			}

			m_ctx->Unmap(m_tmpStagingRes, 0);
		}
	}
}

void CApp::UpdateRMSE()
{
	SShaderCB shaderCB;
	shaderCB.m_imageSizeRcp.x = 1.0f / m_imageWidth;
	shaderCB.m_imageSizeRcp.y = 1.0f / m_imageHeight;
	shaderCB.m_screenSizeRcp.x = 1.0f / m_backbufferWidth;
	shaderCB.m_screenSizeRcp.y = 1.0f / m_backbufferHeight;
	shaderCB.m_texelBias = Vec2(0.0f, 0.0f);
	shaderCB.m_texelScale = 1.0f;
	shaderCB.m_exposure = 1.0f;
	shaderCB.m_blitMode = 0;

	D3D11_MAPPED_SUBRESOURCE mappedRes;
	m_ctx->Map(m_constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedRes);
	memcpy(mappedRes.pData, &shaderCB, sizeof(shaderCB));
	m_ctx->Unmap(m_constantBuffer, 0);
	m_ctx->PSSetConstantBuffers(0, 1, &m_constantBuffer);


	Vec3* imageA = new Vec3[m_imageWidth * m_imageHeight];
	Vec3* imageB = new Vec3[m_imageWidth * m_imageHeight];

	CopyTexture(imageA, m_sourceTextureView);
	CopyTexture(imageB, m_compressedTextureView);
	
	// Compute RGB and Luminance RMSE errors in log space
	double rSum = 0.0;
	double gSum = 0.0;
	double bSum = 0.0;
	for (unsigned y = 0; y < m_imageHeight; ++y)
	{
		for (unsigned x = 0; x < m_imageWidth; ++x)
		{
			double x0 = imageA[x + y * m_imageWidth].x;
			double y0 = imageA[x + y * m_imageWidth].y;
			double z0 = imageA[x + y * m_imageWidth].z;
			double x1 = imageB[x + y * m_imageWidth].x;
			double y1 = imageB[x + y * m_imageWidth].y;
			double z1 = imageB[x + y * m_imageWidth].z;

			double dx = log(x1 + 1.0) - log(x0 + 1.0);
			double dy = log(y1 + 1.0) - log(y0 + 1.0);
			double dz = log(z1 + 1.0) - log(z0 + 1.0);
			rSum += dx * dx;
			gSum += dy * dy;
			bSum += dy * dy;
		}
	}
	m_rgbRMSLE = (float)sqrt((rSum + gSum + bSum) / (3.0 * m_imageWidth * m_imageHeight));
	m_lumRMSLE = (float)sqrt((0.299 * rSum + 0.587 * gSum + 0.114 * bSum) / (1.0 * m_imageWidth * m_imageHeight));

	delete imageA;
	delete imageB;

	char rmseString[256];
	rmseString[0] = 0;
	sprintf_s(rmseString, "rgbRMSLE:%.4f lumRMSLE:%.4f Mode:%s %s\n", m_rgbRMSLE, m_lumRMSLE, m_compressionMode == 1 ? "Quality" : "Fast", ImagePathArr[m_imageID]);
	OutputDebugStringA(rmseString);

	m_updateTitle = true;
}
