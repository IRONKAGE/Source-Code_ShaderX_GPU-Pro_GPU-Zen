#ifndef _CudaHelpers_h_
#define _CudaHelpers_h_

#include "CudaMath.h"


#ifdef __DEVICE_EMULATION__
  #define DEVICE_EMULATION_SYNC() __syncthreads()
  #define DEVICE_EMU_NOINLINE __noinline__
#else // __DEVICE_EMULATION__
  #define DEVICE_EMULATION_SYNC() 
  #define DEVICE_EMU_NOINLINE 
#endif // __DEVICE_EMULATION__




#ifdef _DEBUG

#define CHECK_CUDA_ERROR( ) do \
  {                                      \
  cudaError err = cudaThreadSynchronize();                                 \
  if ( cudaSuccess != err)  \
  {                                               \
    outputFailure(__FILE__, __LINE__, cudaGetErrorString( err));              \
  } } while (0)

#else // _DEBUG
  #define CHECK_CUDA_ERROR( )
#endif // _DEBUG



CUDA_DEVICE_PREAMBLE uint32_t warpReduce(uint32_t data, uint32_t index, volatile uint32_t *sdata)
{
  unsigned int tid = index;
  sdata[tid] = data;

#ifdef __DEVICE_EMULATION__
  __syncthreads();
  if (tid < 16)
    sdata[tid] += sdata[tid + 16]; 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] += sdata[tid +  8]; 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] += sdata[tid +  4]; 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] += sdata[tid +  2]; 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] += sdata[tid +  1]; 
  __syncthreads();
#else // !__DEVICE_EMULATION__
  if (tid < 16)
  {
    sdata[tid] += sdata[tid + 16]; 
    sdata[tid] += sdata[tid +  8]; 
    sdata[tid] += sdata[tid +  4]; 
    sdata[tid] += sdata[tid +  2]; 
    sdata[tid] += sdata[tid +  1]; 
  }
#endif // __DEVICE_EMULATION__
  return sdata[0];
}




template <typename T>
CUDA_DEVICE_PREAMBLE T warpReduceMin(T data, uint32_t index, volatile T *sdata)
{
  unsigned int tid = index;
  sdata[tid] = data;

#ifdef __DEVICE_EMULATION__
  __syncthreads();
  if (tid < 16)
    sdata[tid] = min(sdata[tid], sdata[tid + 16]); 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] = min(sdata[tid], sdata[tid +  8]); 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] = min(sdata[tid], sdata[tid +  4]); 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] = min(sdata[tid], sdata[tid +  2]); 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] = min(sdata[tid], sdata[tid +  1]); 
  __syncthreads();
#else // !__DEVICE_EMULATION__
  if (tid < 16)
  {
    sdata[tid] = min(sdata[tid], sdata[tid + 16]); 
    sdata[tid] = min(sdata[tid], sdata[tid +  8]); 
    sdata[tid] = min(sdata[tid], sdata[tid +  4]); 
    sdata[tid] = min(sdata[tid], sdata[tid +  2]); 
    sdata[tid] = min(sdata[tid], sdata[tid +  1]); 
  }
#endif // __DEVICE_EMULATION__
  return sdata[0];
}




template <typename T>
CUDA_DEVICE_PREAMBLE T warpReduceMax(T data, uint32_t index, volatile T *sdata)
{
  unsigned int tid = index;
  sdata[tid] = data;

#ifdef __DEVICE_EMULATION__
  __syncthreads();
  if (tid < 16)
    sdata[tid] = max(sdata[tid], sdata[tid + 16]); 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] = max(sdata[tid], sdata[tid +  8]); 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] = max(sdata[tid], sdata[tid +  4]); 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] = max(sdata[tid], sdata[tid +  2]); 
  __syncthreads(); 
  if (tid < 16)
    sdata[tid] = max(sdata[tid], sdata[tid +  1]); 
  __syncthreads();
#else // !__DEVICE_EMULATION__
  if (tid < 16)
  {
    sdata[tid] = max(sdata[tid], sdata[tid + 16]); 
    sdata[tid] = max(sdata[tid], sdata[tid +  8]); 
    sdata[tid] = max(sdata[tid], sdata[tid +  4]); 
    sdata[tid] = max(sdata[tid], sdata[tid +  2]); 
    sdata[tid] = max(sdata[tid], sdata[tid +  1]); 
  }
#endif // __DEVICE_EMULATION__
  return sdata[0];
}



// buffer must be +16 elements (i.e. 48), also writes end result to shared, allowing read back of total sum
uint32_t CUDA_DEVICE_PREAMBLE warpPrefixSum3(uint32_t count, uint32_t thid, volatile uint32_t *buffer)
{
  // sets all to 0
  buffer[thid] = 0;
  DEVICE_EMULATION_SYNC();
  //volatile uint32_t *buffer = _buffer + 16;
  thid += 16;
  // Cache the computational window in shared memory
  buffer[thid] = count;
  DEVICE_EMULATION_SYNC();
  //printf("[%d] - %d\n", thid, buffer[thid]);
  DEVICE_EMULATION_SYNC();
  uint32_t lastValue = count;

#ifdef __DEVICE_EMULATION__

#define LOOP_BODY(offset) \
    lastValue += buffer[thid - offset]; \
    DEVICE_EMULATION_SYNC(); \
    buffer[thid] = lastValue; \
    DEVICE_EMULATION_SYNC();
    //printf("[%d] - %d\n", thid, buffer[thid]);


#else // !__DEVICE_EMULATION__

#define LOOP_BODY(offset) \
    lastValue += buffer[thid - offset]; \
    buffer[thid] = lastValue;

#endif // __DEVICE_EMULATION__

  LOOP_BODY(1)
  LOOP_BODY(2)
  LOOP_BODY(4)
  LOOP_BODY(8)
  LOOP_BODY(16)

#undef LOOP_BODY

  DEVICE_EMULATION_SYNC();
  return lastValue - count;
}



template <typename T, uint32_t STRIDE>
void __global__ megaMemsetKernel(T *data, T value, const uint32_t count)
{
# if defined(__CUDACC__)
  for (uint32_t index = threadIdx.x + blockIdx.x * blockDim.x; index < count; index += STRIDE)
  {
    data[index] = value;
  }
# endif // __CUDACC__
}
//
//template <typename T>
//void megaMemset(T *ptr, T value, size_t size)
//{
//  megaMemsetKernel<T, 120 * 256><<<120, 256>>>(ptr, value, uint32_t(size));
//}

void megaMemset(uint32_t *ptr, uint32_t value, size_t size);



__device__ inline uint32_t lowPrecMul(uint32_t a, uint32_t b)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ < 200
	return __umul24(a, b);
#else // __CUDA_ARCH__ >= 200
	return a * b;
#endif // __CUDA_ARCH__ < 200
}



CUDA_HOST_DEVICE_PREAMBLE uint32_t spreadBits(uint32_t value, uint32_t bits, uint32_t stride, uint32_t offset)
{
  uint32_t x = (uint32_t(1) << bits) - 1;
  uint32_t v = value & x;
  uint32_t mask = 1;
  uint32_t result = 0;
  for (uint32 i = 0; i < bits; ++i)
  {
    result |= mask & v;
    v = v << (stride - 1);
    mask = mask << stride;
  }
  return result << offset;
}


template <int BITS>
CUDA_HOST_DEVICE_PREAMBLE uint32_t spreadBits(uint32_t value, uint32_t stride, uint32_t offset)
{
  uint32_t x = (uint32_t(1) << BITS) - 1;
  uint32_t v = value & x;
  uint32_t mask = 1;
  uint32_t result = 0;
  for (uint32 i = 0; i < BITS; ++i)
  {
    result |= mask & v;
    v = v << (stride - 1);
    mask = mask << stride;
  }
  return result << offset;
}

#if defined(__CUDACC__)
__constant__ uint32_t g_mortonLut_3_10bit[1024];
#endif // defined(__CUDACC__)

CUDA_HOST_DEVICE_PREAMBLE uint32_t spreadBits_3_10bit(uint32_t value, uint32_t offset)
{
  uint32_t x = (uint32_t(1) << 10) - 1;
  uint32_t v = value & x;
#if defined(__CUDACC__)
	return g_mortonLut_3_10bit[v] << offset;
#else
	value = (value | (value << 10UL)) & 0x000F801FUL;
	value = (value | (value <<  4UL)) & 0x00E181C3UL;
	value = (value | (value <<  2UL)) & 0x03248649UL;
	value = (value | (value <<  2UL)) & 0x09249249UL;

	return value << offset;
#endif // defined(__CUDACC__)
}



template <int BITS>
CUDA_HOST_DEVICE_PREAMBLE uint32_t unspreadBits(uint32_t value, uint32_t stride, uint32_t offset)
{
  uint32_t v = value >> offset;
  uint32_t mask = 1;
  uint32_t result = 0;
  for (uint32 i = 0; i < BITS; ++i)
  {
    result |= mask & v;
    v = v >> (stride - 1);
    mask = mask << 1;
  }
  return result;
}


#endif // _CudaHelpers_h_
