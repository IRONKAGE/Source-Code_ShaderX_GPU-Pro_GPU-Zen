#ifndef _CudaTexture_h_
#define _CudaTexture_h_

#include "CudaResource.h"

#include "../IntTypes.h"
#include <vector>
#include <driver_types.h>

struct cudaGraphicsResource;
struct cudaArray;

namespace chag
{
/**
 * Interface to texture capable of being mapped to CUDA, may be implemented for any given rendering Api
 * and bound or something I dunno... maybe all that is implementation specific.
 * Prototype implemented for opengl at any rate...
 * Note:
 *  - Creating a texture is hard work as it can be from FBOs or various file formats etc, so better let that be an 
 *    implementation detail, passed in init().
 *  - The cuda array could be replaced with a reference counted smart pointer/object that can be used to ensure all
 *    references are released (in debug mode) before unmapping (this could be something of a hassle though...)
 *  - Stream management.
 *  - 
 */
class CudaTexture : public CudaBaseResource
{
public:
  CudaTexture(uint32_t glTextureId, cudaGraphicsRegisterFlags regFlags);

  /**
   * registers the resource with cuda...
   */
  void init(uint32_t glTextureId);
  void deinit();

  /**
   * Map resource for use with cuda, returns the pointer to the cudaArray associated with mip level 0 by default.
   * unmap() must always be called at some point later (and at the very least before next call to map()).
   */
  cudaArray *map(/* stream */);
  /**
   * Retrieve a pointer to an array for a given mip level, may be called between map/unmap.
   */
  cudaArray *getMipLevel(uint32_t mipLevel);
  /**
   */
  void unmap(/* stream */);

protected:
  uint32_t m_glTextureId;
};


}; // namespace chag


#endif // _CudaTexture_h_
