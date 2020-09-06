#ifndef _CudaGlobals_h_
#define _CudaGlobals_h_

#include "Config.h"
#include <cstring>

#undef near
#undef far

/**
* Common globals needed in many kernels...
*/
struct CudaGlobals
{
	float viewMatrix[16];
	float projectionMatrix[16];

	float inverseView[16];
	float inverseProjectionMatrix[16];
	float inverseViewProjection[16];

	uint32_t fbWidth;
	uint32_t fbHeight;
	uint32_t fbSize;
	uint32_t numMsaaSamples;

	float fbWidthf;
	float fbHeightf;
	float invFbWidth;
	float invFbHeight;
	float invFbSize;

	float near;
	float far;
	float fov;
	float aspectRatio;

	void update(float _projectionMatrix[16], float _inverseProjectionMatrix[16], float _viewMatrix[16], float _inverseView[16],
		float _inverseViewProjection[16], uint32_t _fbWidth, uint32_t _fbHeight, uint32_t _numMsaaSamples, float _near, float _far, float _fov, float _aspectRatio )
	{
		memcpy(projectionMatrix, _projectionMatrix, sizeof(projectionMatrix));
		memcpy(inverseProjectionMatrix, _inverseProjectionMatrix, sizeof(inverseProjectionMatrix));
		memcpy(viewMatrix, _viewMatrix, sizeof(viewMatrix));
		memcpy(inverseView, _inverseView, sizeof(inverseView) );
		memcpy(inverseViewProjection, _inverseViewProjection, sizeof(inverseViewProjection) );

		fbWidth = _fbWidth;
		fbHeight = _fbHeight;
		fbSize = fbWidth * fbHeight;
		numMsaaSamples = _numMsaaSamples;
		fbWidthf = float(_fbWidth);
		fbHeightf = float(_fbHeight);
		invFbWidth = 1.0f / fbWidthf;
		invFbHeight = 1.0f / fbHeightf;
		invFbSize = 1.0f / (fbWidthf * fbHeightf);
		near = _near;
		far = _far;
		fov = _fov;
		aspectRatio = _aspectRatio;
	}
};

#ifdef __CUDACC__

__constant__ CudaGlobals g_globals;


#define CUDA_UPLOAD_GLOBALS(_c_) \
{ \
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_globals, &(_c_), sizeof(CudaGlobals), 0, cudaMemcpyHostToDevice)); \
}

#endif // __CUDACC__

#endif // _CudaGlobals_h_
