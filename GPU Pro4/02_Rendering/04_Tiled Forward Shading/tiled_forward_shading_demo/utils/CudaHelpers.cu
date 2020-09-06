#include "CudaHelpers.h"
#include "CudaCheckError.h"


void megaMemset(uint32_t *ptr, uint32_t value, size_t size)
{
	CUDA_CHECK_ERROR("megaMemset_pre");
  if ((size & 1) == 0)
  {
    megaMemsetKernel<uint2, 120 * 256><<<120, 256>>>(reinterpret_cast<uint2*>(ptr), make_uint2(value, value), uint32_t(size / 2));
  }
  else
  {
    megaMemsetKernel<uint32_t, 120 * 256><<<120, 256>>>(ptr, value, uint32_t(size));
  }
	CUDA_CHECK_ERROR("megaMemset_post");
}