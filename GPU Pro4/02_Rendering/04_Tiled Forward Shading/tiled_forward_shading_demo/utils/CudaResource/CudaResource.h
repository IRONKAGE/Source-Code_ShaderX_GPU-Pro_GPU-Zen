#ifndef _CudaResource_h_
#define _CudaResource_h_

#include "../IntTypes.h"
#include <vector>

struct cudaGraphicsResource;

namespace chag
{

class CudaBaseResource
{
public:
  CudaBaseResource() : m_cudaResource(0), m_mapped(false) { }
  virtual ~CudaBaseResource();

	bool isMapped() const { return m_mapped; }

protected:
  friend class CudaResourceBatchMapper;

  cudaGraphicsResource* m_cudaResource;
  bool m_mapped;
};



/**
 * either use ctor/dtor or use map/unmap, or some combination.
 */
class CudaResourceBatchMapper
{
public:
  /**
   * Will immedietaly map resources.
   */
  CudaResourceBatchMapper(CudaBaseResource **resources = 0, int count = 0);
  /**
   * Will unmap resources, if they have not been already by a explicit call to unmap().
   */
  ~CudaResourceBatchMapper();

  void map(CudaBaseResource **resources, int count);
  void unmap();
protected:
  std::vector<cudaGraphicsResource *> m_resourcePtrs;
  std::vector<CudaBaseResource *> m_resources;
};



}; // namespace chag


#endif // _CudaResource_h_
