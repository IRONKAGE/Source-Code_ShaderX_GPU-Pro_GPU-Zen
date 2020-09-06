#ifndef CUDASAFECALL_H_1A940CAE_ADB2_484F_8C5A_DBD71BD3BF95
#define CUDASAFECALL_H_1A940CAE_ADB2_484F_8C5A_DBD71BD3BF95

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime_api.h>
#include "PlatformCompat.h"

// Avoid dependency on (broken) cutil.h
#define CUDA_SAFE_CALL(call) BEGIN_WARNING_CLOBBER_MSVC do {               \
		cudaError_t err = call;                                                \
		if( cudaSuccess != err ) {                                             \
			fprintf( stderr, "CUDA ERROR `%s'\n", cudaGetErrorString(err) );   \
			fprintf( stderr, "  - in `%s'\n", #call );                         \
			fprintf( stderr, "  - in '%s' on line %d\n", __FILE__, __LINE__ ); \
			abort();                                                           \
		}                                                                      \
	} while(0) BEGIN_WARNING_CLOBBER_MSVC                                   \
	/*ENDM*/

// And cutil_inline.h
#define cutilSafeCall(err) super_cudaSafeCall(err, __FILE__, __LINE__)

inline void super_cudaSafeCall( cudaError_t err, const char* file, const int line )
{
	if( cudaSuccess != err ) 
	{
		fprintf( stderr, "Cuda Error `%s'\n", cudaGetErrorString(err) );
		fprintf( stderr, "  - in '%s' on line %d\n", file, line );
		abort();
	}
}

#endif // CUDASAFECALL_H_1A940CAE_ADB2_484F_8C5A_DBD71BD3BF95
