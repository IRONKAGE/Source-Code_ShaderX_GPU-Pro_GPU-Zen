#include "Config.h"

#if ENABLE_CUDA_OPT_TILED_DEFERRED

#include "DemoTypes.h"
#include "linmath/int2.h"
#include "linmath/float4.h"
#include "CudaGlobals.h"
#include "utils/GlBufferObject.h"

/**
 * This contains all cuda implementations for tiling and shading. It should exist in a single instance
 * and only be created if CUDA is present on the system, if not certain algorithms will not work or fall back to
 * CPU versions.
 * Supports:
 *  Compute shading using tiled deferred method.
 */
class CudaRenderer
{
public:

	virtual void init(uint32_t renderTargetBuffers[DRTI_Max], GlBufferObject<chag::float4> &lightPositionRangeBuffer, 
	  GlBufferObject<chag::float4> &lightColorBuffer, uint32_t resultTexture) = 0;

	virtual void computeTiledShading(const CudaGlobals &globals, uint32_t numLights) = 0;

	virtual void registerStuff(uint32_t resultTexture) = 0;
	virtual void unregisterStuff(uint32_t resultTexture) = 0;

	static CudaRenderer *create();
};

#endif // ENABLE_CUDA_OPT_TILED_DEFERRED
