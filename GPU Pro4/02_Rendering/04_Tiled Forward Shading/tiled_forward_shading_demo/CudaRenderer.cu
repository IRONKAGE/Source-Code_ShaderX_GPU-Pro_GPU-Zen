#include "CudaRenderer.h"

#if ENABLE_CUDA_OPT_TILED_DEFERRED

#include "utils/CudaHelpers.h"
#include "utils/CudaResource/CudaTexture.h"
#include "utils/CudaResource/CudaBuffer.h"
#include "utils/CudaBuffer.h"
#include "utils/CudaCheckError.h"
#include <cuda_gl_interop.h>
#include <performance_monitor/profiler/Profiler.h>

#undef near
#undef far



class CudaRendererImpl : public CudaRenderer
{
public:

	virtual void init(uint32_t renderTargetBuffers[DRTI_Max], GlBufferObject<chag::float4> &lightPositionRangeBuffer, 
	                  GlBufferObject<chag::float4> &lightColorBuffer, uint32_t resultTexture);

	virtual void computeTiledShading(const CudaGlobals &globals, uint32_t numLights);

	virtual void registerStuff(uint32_t resultTexture)
	{
		ASSERT(!m_resultTexture);
		m_resultTexture = new chag::CudaTexture(resultTexture, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	}

	virtual void unregisterStuff(uint32_t resultTexture)
	{
		if (m_resultTexture)
		{
			delete m_resultTexture;
		}
		m_resultTexture = 0;
	}

protected:
	CudaGlobals m_globals;

	// Resources that are mapped from OpenGL.
	// deferred targets (G-Buffers):
	chag::CudaBufferObject<const void> *m_diffuseBuffer;
	chag::CudaBufferObject<const void> *m_specularShininessBuffer;
	chag::CudaBufferObject<const void> *m_normalBuffer;
	chag::CudaBufferObject<const void> *m_ambientBuffer;
	chag::CudaBufferObject<const float> *m_depthBuffer;
	chag::CudaTexture *m_resultTexture;

	chag::CudaBufferObject<const chag::float4> *m_lightPositionRangeBuffer;
	chag::CudaBufferObject<const chag::float4> *m_lightColorBuffer;
};



CudaRenderer *CudaRenderer::create()
{
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) == cudaSuccess)
	{
		printf("CUDA detected, initializing\n");
		int cudaDevice = 0; // TODO -- which device should we use?
		cutilSafeCall(cudaGLSetGLDevice(cudaDevice));
		CUT_CHECK_ERROR("cudaGLSetGLDevice");
		Profiler::setEnabledTimers(Profiler::TT_All);

		return new CudaRendererImpl;
	}
	else
	{
		printf("NO CUDA device/driver detected, CPU fallback will be used.\n");
		Profiler::setEnabledTimers(Profiler::TT_Cpu | Profiler::TT_OpenGl);
	}

	return 0;
}


void CudaRendererImpl::init(uint32_t renderTargetBuffers[DRTI_Max], GlBufferObject<chag::float4> &lightPositionRangeBuffer, 
	                          GlBufferObject<chag::float4> &lightColorBuffer, uint32_t resultTexture)
{
	m_diffuseBuffer = new chag::CudaBufferObject<const void>(renderTargetBuffers[DRTI_Diffuse], cudaGraphicsRegisterFlagsReadOnly);
	m_specularShininessBuffer = new chag::CudaBufferObject<const void>(renderTargetBuffers[DRTI_SpecularShininess], cudaGraphicsRegisterFlagsReadOnly);
	m_normalBuffer = new chag::CudaBufferObject<const void>(renderTargetBuffers[DRTI_Normal], cudaGraphicsRegisterFlagsReadOnly);
	m_ambientBuffer = new chag::CudaBufferObject<const void>(renderTargetBuffers[DRTI_Ambient], cudaGraphicsRegisterFlagsReadOnly);
	m_depthBuffer = new chag::CudaBufferObject<const float>(renderTargetBuffers[DRTI_Depth], cudaGraphicsRegisterFlagsReadOnly);
	m_resultTexture = new chag::CudaTexture(resultTexture, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	m_lightPositionRangeBuffer = new chag::CudaBufferObject<const chag::float4>(lightPositionRangeBuffer, cudaGraphicsRegisterFlagsReadOnly);
	m_lightColorBuffer = new chag::CudaBufferObject<const chag::float4>(lightColorBuffer, cudaGraphicsRegisterFlagsReadOnly);
}






inline __device__ float4 transform(float m[16], const float4& v) 
{
  return make_float4(m[0] * v.x + m[4] * v.y + m[8]  * v.z + m[12] * v.w, 
				             m[1] * v.x + m[5] * v.y + m[9]  * v.z + m[13] * v.w, 
				             m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
                     m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w);
}



__device__ float3 unProject(float2 fragmentPos, float fragmentDepth)
{
  float4 pt = transform(g_globals.inverseProjectionMatrix, make_float4(fragmentPos.x * 2.0f - 1.0f, fragmentPos.y * 2.0f - 1.0f, 2.0f * fragmentDepth - 1.0f, 1.0f));
  float3 pp = make_float3(pt.x, pt.y, pt.z) / pt.w;

  return pp;
}



__device__ float3 fetchPosition(uint2 p, uint32_t sampleOffset, const float *depthBuffer)
{
  float2 fragmentPos = make_float2(float(p.x) * g_globals.invFbWidth, float(p.y) * g_globals.invFbHeight);
  float d = depthBuffer[sampleOffset];
  return unProject(fragmentPos, d);
}



__constant__ uint32_t g_numLights;

surface<void, 2> g_resultSurface;

texture<float4, 1, cudaReadModeElementType> g_diffuseTex;
texture<float4, 1, cudaReadModeElementType> g_normalTex;
texture<float4, 1, cudaReadModeElementType> g_ambientTex;
texture<float4, 1, cudaReadModeElementType> g_specularShininessTex;

texture<float4, 1, cudaReadModeElementType> g_light_position_range_tex;
texture<float4, 1, cudaReadModeElementType> g_light_color_tex;



__device__ float screenToViewDepth(float screenDepth)
{
  float zDepth = screenDepth * 2.0f - 1.0f;
  // flip view Z to be positive as we use it for integer atomic comparison below.
  return 2.0f * g_globals.near * g_globals.far / (zDepth * (g_globals.far - g_globals.near) - (g_globals.far + g_globals.near));
}


__device__ uchar4 packPixelColor(float3 c)
{
	c = saturate(c);
	uchar4 r = { uint8_t(c.x * 255.0f), uint8_t(c.y * 255.0f), uint8_t(c.z * 255.0f), 255U };
  return r;
}



__device__ float3 unpackPixelColor(uchar4 c)
{
	float3 r = { float(c.x), float(c.y), float(c.z) };
  return r / 255.0f;
}



__device__ void doLightOpt(float3 position, float3 normal, float shininess, float3 viewDir, float3 lightPos, float3 color, float range, float3 &diffLight, float3 &specLight)
{
  float3 lightDir = float3(lightPos) - position;
  float dist = length3(lightDir);
  lightDir *= 1.0f / dist;
  float inner = 0.0f;

  float spec = powf(max(0.0f, dot3(normalize3(lightDir + viewDir), normal)), shininess);

	// TODO: This does not do the fresnel refl etc.

  float ndotL = max(dot3(normal, lightDir),0.0f);

  float att =  max(1.0f - max(0.0f, (dist - inner) / (range - inner)), 0.0f); // dist < radius ? 1.0f : 0.0f;
  specLight += color * att * spec * ndotL;
  diffLight += color * (att * ndotL);
}



__device__ void doLightOpt(float3 position, float3 normal, float shininess, float3 viewDir, int lightIndex, float3 &diffLight, float3 &specLight)
{
  float3 lightColor = make_float3(tex1Dfetch(g_light_color_tex, lightIndex));
  float4 tmp = tex1Dfetch(g_light_position_range_tex, lightIndex);
  float3 lightPos = make_float3(tmp);
  float lightRange = tmp.w;

	doLightOpt(position, normal, shininess, viewDir, lightPos, lightColor, lightRange, diffLight, specLight);
}



#define SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_X 32
#define SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_Y 8



__device__ float3 shadeSample(uint2 p, uint32_t sampleOffset, const float * __restrict__ depthBuffer, uint32_t tileNumLights, const uint32_t *tileLightIndices)
{
	const float3 position = fetchPosition(p, sampleOffset, depthBuffer);
	const float4 specularShininess = tex1Dfetch(g_specularShininessTex, sampleOffset);
	const float3 normal = make_float3(tex1Dfetch(g_normalTex, sampleOffset));
	const float3 specular = make_float3(specularShininess);
	float shininess = specularShininess.w;
	const float3 viewDir = -normalize3(position);

	float3 diffLight = make_float3(0.0f);
	float3 specLight = make_float3(0.0f);
	// copute lighting for tile lights.
	for (uint32_t l = 0; l < tileNumLights; ++l)
	{
		uint32_t lightIndex = tileLightIndices[l];
		doLightOpt(position, normal, shininess, viewDir, lightIndex, diffLight, specLight);
	}

	const float3 diffuse = make_float3(tex1Dfetch(g_diffuseTex, sampleOffset));

	return diffLight * diffuse + specLight * specular + make_float3(tex1Dfetch(g_ambientTex, sampleOffset));
}



__global__ void computeTiledDeferredSingleKernel(const float *__restrict__ depthBuffer)
{
  uint32_t groupIndex = threadIdx.y * SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_X + threadIdx.x;

  uint2 tileIndex = make_uint2(blockIdx.x, blockIdx.y);
  uint2 p = make_uint2(blockIdx.x * SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_X + threadIdx.x, blockIdx.y * SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_Y + threadIdx.y);

  __shared__ uint32_t sMinZ;
  __shared__ uint32_t sMaxZ;

  // Light list for the tile
  __shared__ uint32_t sTileLightIndices[NUM_POSSIBLE_LIGHTS];
  __shared__ uint32_t sTileNumLights;


  __shared__ uint32_t sNumPerSamplePixels;
  __shared__ uint32_t sPerSamplePixels[SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_X * SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_Y];

  // Initialize shared memory light list and Z bounds
  if (groupIndex == 0) 
  {
    sTileNumLights = 0;
		sNumPerSamplePixels = 0;
    sMinZ = 0x7F7FFFFF;      // Max float
    sMaxZ = 0;
  }

	bool sampleOnScreen = p.x < g_globals.fbWidth && p.y < g_globals.fbHeight;

  __syncthreads();

	uint32_t firstSampleOffset = (p.y * g_globals.fbWidth + p.x) * g_globals.numMsaaSamples;

	// local depth min and max, in screen space, i.e. 0-1, with 0 nearest.
	float depthMin = 1.0f;
	float depthMax = 0.0f;//depthMin;
	
	if (sampleOnScreen)
	{
		for (uint32_t i = 0; i < g_globals.numMsaaSamples; ++i)
		{
			float depth = depthBuffer[firstSampleOffset + i];
			if (depth < 1.0f)
			{
				depthMin = min(depthMin, depth);
				depthMax = max(depthMax, depth);
			}
		}
	}

	// then reduce min and max for tile:

  // Use warp reduction first and then use atomics to reduce between warps, should ensure 0 contention and only
  // one sync... is plenty faster anyways.
  uint32_t threadInWarpIndex = threadIdx.x;
  uint32_t warpIndex = threadIdx.y;

  uint32_t warpMin = warpReduceMin(uint32_t(float_as_int(depthMin)), threadInWarpIndex, &sTileLightIndices[warpIndex * 32]);
  if (threadInWarpIndex == 0)
  {
    atomicMin(&sMinZ, warpMin);
  }

  uint32_t warpMax = warpReduceMax(uint32_t(float_as_int(depthMax)), threadInWarpIndex, &sTileLightIndices[warpIndex * 32]);
  if (threadInWarpIndex == 0)
  {
    atomicMax(&sMaxZ, warpMax);
  }
  __syncthreads();


  float minTileZ = -screenToViewDepth(int_as_float(sMinZ));
  float maxTileZ = -screenToViewDepth(int_as_float(sMaxZ));

  // if turned off, then light culling and lighting is disabled, and the min/max diff is output as result. 
#define ENABLE_LIGHT_CULLING_AND_LIGHTING 1
#if ENABLE_LIGHT_CULLING_AND_LIGHTING
  // NOTE: This is all uniform per-tile (i.e. no need to do it per-thread) but fairly inexpensive
  // We could just precompute the frusta planes for each tile and dump them into a constant buffer...
  // They don't change unless the projection matrix changes since we're doing it in view space.
  // Then we only need to compute the near/far ones here tightened to our actual geometry.
  // The overhead of group synchronization/LDS or global memory lookup is probably as much as this
  // little bit of math anyways, but worth testing.

  // Work out scale/bias from [0, 1]
//  float2 tileScale = make_float2(g_globals.fbWidthf, g_globals.fbHeightf) / float(2 * SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_X);
//  float2 tileBias = tileScale - make_float2(tileIndex);

  float2 tileScale = make_float2(g_globals.fbWidthf, g_globals.fbHeightf) / make_float2(SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_X, SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_Y);
  // note though, that the 'bias' here already has some of the matrix composition work rolled in.
  float2 tileBias = make_float2(tileIndex) * 2.0f - tileScale + 1.0f;

  // Now work out composite (i.e. biased and scaled) projection matrix
  // Relevant matrix columns for this tile frusta
  float4 c1 = make_float4(g_globals.projectionMatrix[0] * tileScale.x, 0.0f, tileBias.x, 0.0f);
  float4 c2 = make_float4(0.0f, g_globals.projectionMatrix[1 + 1 * 4] * tileScale.y, tileBias.y, 0.0f);
  float4 c4 = make_float4(0.0f, 0.0f, -1.0f, 0.0f);

  // Derive frustum planes
  float4 frustumPlanes[6];
  // Sides
  frustumPlanes[0] = c4 - c1;
  frustumPlanes[1] = c4 + c1;
  frustumPlanes[2] = c4 - c2;
  frustumPlanes[3] = c4 + c2;
  // Near/far
  frustumPlanes[4] = make_float4(0.0f, 0.0f, -1.0f, -minTileZ);
  frustumPlanes[5] = make_float4(0.0f, 0.0f,  1.0f,  maxTileZ);

  // Normalize frustum planes (near/far already normalized)
  #pragma unroll
  for (uint32_t i = 0; i < 4; ++i)
  {
    frustumPlanes[i] *= rlength3(frustumPlanes[i]);
  }

  // Cull lights for this tile
  for (uint32_t lightIndex = groupIndex; lightIndex < g_numLights; lightIndex += (SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_X * SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_Y)) 
  {
    float4 lightPosRange = tex1Dfetch(g_light_position_range_tex, lightIndex); //g_light_position_range[lightIndex];

    // Cull: point light sphere vs tile frustum
    bool inFrustum = true;
#if 1
    // manual unroll as the compiler refused (advisory warning)...
    // should make a big difference as a lot of 0 and 1 in these planes... (it did, according to measurements:
    //  got 5 ms quicker for even more lights scene).
#define LOOP_BODY(_i_) \
    { \
      float d = dot4(frustumPlanes[_i_], make_float4(make_float3(lightPosRange), 1.0f));  \
      inFrustum = inFrustum && (d >= -lightPosRange.w); \
    }
    LOOP_BODY(0);
    LOOP_BODY(1);
    LOOP_BODY(2);
    LOOP_BODY(3);
    LOOP_BODY(4);
    LOOP_BODY(5);

#undef LOOP_BODY

#else 
    #pragma unroll
    for (uint32_t i = 0; i < 6 ; ++i) 
    {
      float d = dot4(frustumPlanes[i], make_float4(make_float3(lightPosRange), 1.0f));
      inFrustum = inFrustum && (d >= -lightPosRange.w);
    }
#endif // 0

    if (inFrustum) 
    {
      // Append light to list
      // Compaction might be better if we expect a lot of lights
      uint32_t listIndex = atomicAdd(&sTileNumLights, 1);
      sTileLightIndices[listIndex] = lightIndex;
    }
  }
  __syncthreads();

#if ENABLE_FRAGMENT_COUNTER
  // do once per tile...
  if (groupIndex == 0) 
  {
    atomicAdd(&g_totalLightsInTiles, sTileNumLights);
  }
#endif // ENABLE_FRAGMENT_COUNTER

	// NOTE: not returning here, as we need all threads awake to shade samples later...
	if (sampleOnScreen)
	{
  // If turned off, the lighting is skipped, and the number of lights is output as result.
#define ENABLE_LIGHTING 1
#if ENABLE_LIGHTING

#define ENABLE_PER_SAMPLE_SHADING_OPT 1

#if ENABLE_PER_SAMPLE_SHADING_OPT
		bool perSampleShading = false;

		//const float maxZDelta = abs(surface[0].positionViewDX.z) + abs(surface[0].positionViewDY.z);
		const float minNormalDot = 0.99f;        // Allow ~8 degree normal deviations
		const float3 normal0 = make_float3(tex1Dfetch(g_normalTex, firstSampleOffset));

		for (uint32_t i = 1; i < g_globals.numMsaaSamples; ++i) 
		{
			// Using the position derivatives of the triangle, check if all of the sample depths
			// could possibly have come from the same triangle/surface

			//perSampleShading = perSampleShading || 
			//	abs(surface[i].positionView.z - surface[0].positionView.z) > maxZDelta;

			// Also flag places where the normal is different
			const float3 normal = make_float3(tex1Dfetch(g_normalTex, firstSampleOffset + i));
			perSampleShading = perSampleShading || dot3(normal, normal0) < minNormalDot;
		}
#else // !ENABLE_PER_SAMPLE_SHADING_OPT
		bool perSampleShading = true;
#endif // ENABLE_PER_SAMPLE_SHADING_OPT

		// always shade 0th sample.
		float3 light = shadeSample(p, firstSampleOffset + 0, depthBuffer, sTileNumLights, sTileLightIndices);

		// shade the rest if needed...
		if (perSampleShading)
		{
#define DEFER_PER_SAMPLE_SHADING 1
#if DEFER_PER_SAMPLE_SHADING
			uint32_t listIndex = atomicAdd(&sNumPerSamplePixels, 1);
			sPerSamplePixels[listIndex] = p.x << 16 | p.y;
#else // !DEFER_PER_SAMPLE_SHADING
			for (uint32_t i = 1; i < g_globals.numMsaaSamples; ++i)
			{
				light += shadeSample(p, firstSampleOffset + i, depthBuffer, sTileNumLights, sTileLightIndices);
			}

			light *= 1.0f / float(g_globals.numMsaaSamples);
#endif // DEFER_PER_SAMPLE_SHADING
		}

		if (perSampleShading)
		{
			//light.x += 0.2f;
		}

#else // !ENABLE_LIGHTING
		// visualize light load.
		float3 light = make_float3(float(sTileNumLights) / float(g_numLights));
#endif // ENABLE_LIGHTING
		// TODO: note that the 4 here assumes 8 bit target texture... 
		surf2Dwrite(packPixelColor(toSrgb(light)), g_resultSurface, p.x * 4, p.y);
	}
#if DEFER_PER_SAMPLE_SHADING
	__syncthreads();
	__threadfence_block();
  for (uint32_t pixelIndex = groupIndex; pixelIndex < sNumPerSamplePixels; pixelIndex += (SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_X * SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_Y)) 
  {
		uint2 p = make_uint2(sPerSamplePixels[pixelIndex] >> 16, sPerSamplePixels[pixelIndex] & ((1U << 16U) - 1));
		uint32_t firstSampleOffset = (p.y * g_globals.fbWidth + p.x) * g_globals.numMsaaSamples;

		uchar4 preColor;
		surf2Dread(&preColor, g_resultSurface, p.x * 4, p.y);
		float3 light = fromSrgb(unpackPixelColor(preColor));
		for (uint32_t i = 1; i < g_globals.numMsaaSamples; ++i)
		{
			light += shadeSample(p, firstSampleOffset + i, depthBuffer, sTileNumLights, sTileLightIndices);
		}

		light *= 1.0f / float(g_globals.numMsaaSamples);
		surf2Dwrite(packPixelColor(toSrgb(light)), g_resultSurface, p.x * 4, p.y);
	}
#endif // DEFER_PER_SAMPLE_SHADING
#else // !ENABLE_LIGHT_CULLING_AND_LIGHTING
  
	if (p.x >= g_globals.fbWidth || p.y >= g_globals.fbHeight)
  {
    return;
  }
  float zRange = maxTileZ - minTileZ;
  float3 r = make_float3((zRange) / (g_globals.far - g_globals.near));
  //float3 r = make_float3((zRange) / (1000.0f), (zRange) / (100.0f), (zRange) / (10.0f));
	//surf2Dwrite(make_float4(r, 1.0f), g_resultSurface, p.x * 4, p.y);
	uchar4 asd = { 128, 0, 255, 128 };
	surf2Dwrite(packPixelColor(r), g_resultSurface, p.x * 4, p.y);
#endif // ENABLE_LIGHT_CULLING_AND_LIGHTING
}


void CudaRendererImpl::computeTiledShading(const CudaGlobals &globals, uint32_t numLights)
{
	m_globals = globals;
	CUDA_UPLOAD_GLOBALS(m_globals);

	chag::CudaBaseResource *resouces[] = 
	{
		m_diffuseBuffer,
		m_specularShininessBuffer,
		m_normalBuffer,
		m_ambientBuffer,
		m_depthBuffer,
		m_lightColorBuffer,
		m_lightPositionRangeBuffer,
		m_resultTexture,
	};

	{
		PROFILE_BEGIN_BLOCK_2("map", TT_Cuda);
		// map resources in this block, auto unmaps at end.
		chag::CudaResourceBatchMapper mappo(resouces, sizeof(resouces) / sizeof(resouces[0]));
		PROFILE_END_BLOCK_2();

		cudaMemcpyToSymbol(g_numLights, &numLights, sizeof(numLights), 0, cudaMemcpyHostToDevice);
		cudaBindSurfaceToArray(g_resultSurface, m_resultTexture->getMipLevel(0));
		cudaChannelFormatDesc desc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
		
		cudaBindTexture(0, &g_diffuseTex, m_diffuseBuffer->getPointer(), &desc, size_t(globals.fbWidth * globals.fbHeight * globals.numMsaaSamples * 8));
		CUDA_CHECK_ERROR("cudaBindTexture g_diffuseTex");
		cudaBindTexture(0, &g_specularShininessTex, m_specularShininessBuffer->getPointer(), &desc, size_t(globals.fbWidth * globals.fbHeight * globals.numMsaaSamples * 8));
		CUDA_CHECK_ERROR("cudaBindTexture g_specularShininessTex");
		cudaBindTexture(0, &g_normalTex, m_normalBuffer->getPointer(), &desc, size_t(globals.fbWidth * globals.fbHeight * globals.numMsaaSamples * 8));
		CUDA_CHECK_ERROR("cudaBindTexture g_normalTex");
		cudaBindTexture(0, &g_ambientTex, m_ambientBuffer->getPointer(), &desc, size_t(globals.fbWidth * globals.fbHeight * globals.numMsaaSamples * 8));
		CUDA_CHECK_ERROR("cudaBindTexture g_ambientTex");

		cudaBindTexture(0, g_light_position_range_tex, reinterpret_cast<const ::float4 *>(m_lightPositionRangeBuffer->getPointer()), numLights * sizeof(float4));
		cudaBindTexture(0, g_light_color_tex, reinterpret_cast<const ::float4 *>(m_lightColorBuffer->getPointer()), numLights * sizeof(float4));

    dim3 tileSize = dim3(SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_X, SINGLE_KERNEL_LIGHT_GRID_TILE_DIM_Y);
    dim3 dimGrid = getBlockCount(tileSize, make_uint2(globals.fbWidth, globals.fbHeight));
		
		{
			PROFILE_SCOPE_2("computeTiledDeferredSingleKernel", TT_Cuda);
			computeTiledDeferredSingleKernel<<<dimGrid, tileSize>>>(m_depthBuffer->getPointer());
			CUDA_CHECK_ERROR("computeTiledDeferredSingleKernel");
		}
	}
}

#endif // ENABLE_CUDA_OPT_TILED_DEFERRED
