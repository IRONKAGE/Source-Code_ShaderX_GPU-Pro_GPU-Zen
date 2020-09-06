#include "CudaBuffer.h"

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "../Assert.h"
#include "../CudaSafeCall.h"

namespace chag
{



CudaBaseBuffer::CudaBaseBuffer(uint32_t glBufferId, cudaGraphicsRegisterFlags regFlags) : 
  m_glBufferId(glBufferId)
{
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_glBufferId, regFlags));
}



void* CudaBaseBuffer::map(/*todo: stream*/)
{
  ASSERT(!m_mapped);
  ASSERT(m_cudaResource);
  void *ptr = 0;
	size_t size = 0;
  if (m_cudaResource && !m_mapped)
  {
    /*do: check errors */ 
    cudaGraphicsMapResources(1, &m_cudaResource);
    m_mapped = true;
    cudaGraphicsResourceGetMappedPointer(&ptr, &size, m_cudaResource);
  }
  return ptr;
}



void *CudaBaseBuffer::getPointer()
{
  ASSERT(m_cudaResource);
  ASSERT(m_mapped); // if non-null then resource is mapped

  void *ptr = 0;
	size_t size = 0;
  if (m_cudaResource && m_mapped)
  {
    cudaGraphicsResourceGetMappedPointer(&ptr, &size, m_cudaResource);
  }
  return ptr;
}



void CudaBaseBuffer::unmap()
{
  ASSERT(m_cudaResource);
  ASSERT(m_mapped);
  m_mapped = false;
  cudaGraphicsUnmapResources(1, &m_cudaResource);
}



}; // namespace chag
