#ifndef CUDACHECKERROR_H_0D229A50_E661_4949_B3FD_7480948BAD7C
#define CUDACHECKERROR_H_0D229A50_E661_4949_B3FD_7480948BAD7C

#include <cstdio>
#include <cstdlib>
#include "PlatformCompat.h"

// Avoid dependency on (broken) cutil.h
#if defined(_DEBUG) || !defined(NDEBUG)
#define CUDA_CHECK_ERROR(msg) BEGIN_WARNING_CLOBBER_MSVC do {               \
	cudaError_t lastErr = cudaGetLastError();                                 \
	if( cudaSuccess != lastErr ) {                                            \
		fprintf( stderr, "CUDA ERROR: `%s'\n", cudaGetErrorString(lastErr) ); \
		fprintf( stderr, "  - in CUDA_CHECK_ERROR() // cudaGetLastError()\n" );\
		fprintf( stderr, "  - in '%s' on line %d\n", __FILE__, __LINE__ );    \
		abort();                                                              \
	}                                                                         \
	                                                                          \
	cudaError_t syncErr = cudaDeviceSynchronize();                            \
	if( cudaSuccess != syncErr ) {                                            \
		fprintf( stderr, "CUDA ERROR: `%s'\n", cudaGetErrorString(syncErr) ); \
		fprintf( stderr, "  - in CUDA_CHECK_ERROR() // cudaDeviceSync()\n" ); \
		fprintf( stderr, "  - in '%s' on line %d\n", __FILE__, __LINE__ );    \
		abort();                                                              \
	}                                                                         \
	} while(0)BEGIN_WARNING_CLOBBER_MSVC                                     \
	/*ENDM*/

#else // !DEBUG
#define CUDA_CHECK_ERROR(msg) BEGIN_WARNING_CLOBBER_MSVC do {                                            \
	cudaError_t lastErr = cudaGetLastError();                                 \
	if( cudaSuccess != lastErr ) {                                            \
		fprintf( stderr, "CUDA ERROR: `%s'\n", cudaGetErrorString(lastErr) ); \
		fprintf( stderr, "  - in CUDA_CHECK_ERROR() // cudaGetLastError()\n" );\
		fprintf( stderr, "  - in '%s' on line %d\n", __FILE__, __LINE__ );    \
		abort();                                                              \
	}                                                                         \
	} while(0) BEGIN_WARNING_CLOBBER_MSVC                                                               \
	/*ENDM*/

#endif // DEBUG

#define CUT_CHECK_ERROR(msg) CUDA_CHECK_ERROR(msg)

#endif // CUDACHECKERROR_H_0D229A50_E661_4949_B3FD_7480948BAD7C
