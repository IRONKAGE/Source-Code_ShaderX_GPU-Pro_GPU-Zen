#ifndef _chag_CudaBufferObject_h_
#define _chag_CudaBufferObject_h_

#include "CudaResource.h"

#include "../IntTypes.h"
#include <vector>
#include <driver_types.h>

struct cudaGraphicsResource;
struct cudaArray;

namespace chag
{
/**
 * Interface to buffer object capable of being mapped to CUDA, may be implemented for any given rendering Api
 * and bound or something I dunno... maybe all that is implementation specific.
 * Prototype implemented for opengl at any rate...
 * Note:
 *  - Stream management.
 */
class CudaBaseBuffer : public CudaBaseResource
{
public:
  CudaBaseBuffer(uint32_t glBufferId, cudaGraphicsRegisterFlags regFlags);

  /**
   * registers the resource with cuda...
   */
  void init(uint32_t glBufferId);
  void deinit();

  /**
   * Map resource for use with cuda, returns the pointer to the cudaArray associated with mip level 0 by default.
   * unmap() must always be called at some point later (and at the very least before next call to map()).
   */
  void *map(/* stream */);
  /**
   * Retrieve a cuda pointer buffer, may be called between map/unmap.
   */
  void *getPointer();

  /**
   */
  void unmap(/* stream */);

protected:
  uint32_t m_glBufferId;
};

template <typename T>
class CudaBufferObject : public CudaBaseBuffer
{
public:
	CudaBufferObject(uint32_t glBufferId, cudaGraphicsRegisterFlags regFlags) : CudaBaseBuffer(glBufferId, regFlags) { }

  /**
   * Map resource for use with cuda, returns the pointer to the cudaArray associated with mip level 0 by default.
   * unmap() must always be called at some point later (and at the very least before next call to map()).
   */
	T *map(/* stream */) 
	{ 
		return reinterpret_cast<T*>(CudaBaseBuffer::map()); 
	}
  /**
   * Retrieve a cuda pointer buffer, may be called between map/unmap.
   */
	T *getPointer() 
	{ 
		return reinterpret_cast<T*>(CudaBaseBuffer::getPointer()); 
	}
};



}; // namespace chag


#endif // _chag_CudaBufferObject_h_
