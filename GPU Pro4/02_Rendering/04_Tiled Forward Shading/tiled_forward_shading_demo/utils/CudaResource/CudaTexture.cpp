#include "CudaTexture.h"

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "../Assert.h"
#include "../CudaSafeCall.h"

namespace chag
{



CudaTexture::CudaTexture(uint32_t glTextureId, cudaGraphicsRegisterFlags regFlags) : 
  m_glTextureId(glTextureId)
{
  cutilSafeCall(cudaGraphicsGLRegisterImage(&m_cudaResource, m_glTextureId, GL_TEXTURE_2D, regFlags));
}



cudaArray* CudaTexture::map(/*todo: stream*/)
{
  ASSERT(!m_mapped);
  ASSERT(m_cudaResource);
  cudaArray *arr = 0;
  if (m_cudaResource && !m_mapped)
  {
    /*do: check errors */ 
    cudaGraphicsMapResources(1, &m_cudaResource);
    m_mapped = true;
    cudaGraphicsSubResourceGetMappedArray(&arr, m_cudaResource, 0, 0);
  }
  return arr;
}



cudaArray *CudaTexture::getMipLevel(uint32_t mipLevel)
{
  ASSERT(m_cudaResource);
  ASSERT(m_mapped); // if non-null then resource is mapped

  cudaArray *arr = 0;
  if (m_cudaResource && m_mapped)
  {
    cudaGraphicsSubResourceGetMappedArray(&arr, m_cudaResource, 0, mipLevel);
  }
  return arr;
}



void CudaTexture::unmap()
{
  ASSERT(m_cudaResource);
  ASSERT(m_mapped);
  m_mapped = false;
  cudaGraphicsUnmapResources(1, &m_cudaResource);
}



}; // namespace chag
