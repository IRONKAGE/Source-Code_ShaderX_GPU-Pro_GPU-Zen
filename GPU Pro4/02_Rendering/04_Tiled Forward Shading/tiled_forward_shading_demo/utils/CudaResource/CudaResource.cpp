#include "CudaResource.h"

#include <cuda_runtime_api.h>

#include "../Assert.h"

namespace chag
{


CudaBaseResource::~CudaBaseResource()
{
  ASSERT(!m_mapped);
  if (m_cudaResource)
  {
    cudaGraphicsUnregisterResource(m_cudaResource);
  }
}



CudaResourceBatchMapper::CudaResourceBatchMapper(CudaBaseResource **resources, int count)
{
  if (resources && count)
  {
    map(resources, count);
  }
}



CudaResourceBatchMapper::~CudaResourceBatchMapper()
{
	if (m_resources.size())
  {
    unmap();
  }
}



void CudaResourceBatchMapper::map(CudaBaseResource **resources, int count)
{
  ASSERT(count);
  ASSERT(resources);

  m_resources.resize(count);
  m_resourcePtrs.resize(count);
  // shovel resource ptrs into array,
  // check/set all bool flags on resources.
  for (int i = 0; i < count; ++i)
  {
    CudaBaseResource *r = resources[i];
    ASSERT(!r->m_mapped);
    r->m_mapped = true;

		m_resources[i] = r;
    m_resourcePtrs[i] = r->m_cudaResource;
  }

	cudaGraphicsMapResources(m_resourcePtrs.size(), &m_resourcePtrs[0]);
}



void CudaResourceBatchMapper::unmap()
{
	ASSERT(!m_resources.empty());
  ASSERT(m_resources.size() == m_resourcePtrs.size());

  // check/clear all bool flags
	for (size_t i = 0; i < m_resources.size(); ++i)
  {
    CudaBaseResource *r = m_resources[i];
    ASSERT(r->m_mapped);
    r->m_mapped = false;
  }

  cudaGraphicsUnmapResources(m_resourcePtrs.size(), &m_resourcePtrs[0]);

  // clear the pointer & count, so destructor will not unmap again.
	m_resourcePtrs.clear();
	m_resources.clear();
}



}; // namespace chag
