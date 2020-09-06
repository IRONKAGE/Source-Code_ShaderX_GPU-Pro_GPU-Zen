#ifndef _CudaMath_h_
#define _CudaMath_h_

#include "IntTypes.h"


/**
 * Contains math definitions that are compatible with the cuda compiler, the aim is to make
 * it simpler to move code between cuda and straight C++ code.
 */

// We must emulate some functionality in cuda if things are to compile
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE_PREAMBLE __inline__ __device__ __host__
#define CUDA_DEVICE_PREAMBLE inline __device__ 

#define BASEMATH_NO_CONSTANTS
#define BASEMATH_FUNC_PREAMBLE CUDA_HOST_DEVICE_PREAMBLE

#define g_halfPi (3.1415926535897932384626433832795f / 2.0f)
//#define g_pi (3.1415926535897932384626433832795f)
#define g_defaultEpsilon (0.000001f)

typedef float4 Float4ArgumentType;
typedef float3 Float3ArgumentType;
typedef float2 Float2ArgumentType;

#else // __CUDACC__

#include <math.h>
#include <float.h>


// enable this before including to avoid any cuda headers, will declare cuda types that otherwise are provided by cuda_runtime.h.
#ifdef CUDA_MATH_NO_CUDA

struct float4
{
  float x,y,z,w;
};

inline float4 make_float4(float x, float y, float z, float w)
{
  float4 r = { x, y, z, w };
  return r;
}

struct float3
{
  float x,y,z;
};

inline float3 make_float3(float x, float y, float z)
{
  float3 r = { x, y, z};
  return r;
}

struct float2
{
  float x,y;
};

inline float2 make_float2(float x, float y)
{
  float2 r = { x, y };
  return r;
}

#else // !CUDA_MATH_NO_CUDA

#include <cuda_runtime.h>

#endif // CUDA_MATH_NO_CUDA

#define CUDA_HOST_DEVICE_PREAMBLE inline
#define CUDA_DEVICE_PREAMBLE inline

typedef const float4 &Float4ArgumentType;
typedef const float3 &Float3ArgumentType;
typedef const float2 &Float2ArgumentType;

template <typename T>
inline T max(T a, T b)
{
  return a > b ? a : b;
}

template <typename T>
inline T min(T a, T b)
{
  return a < b ? a : b;
}

const float g_halfPi = 3.1415926535897932384626433832795f / 2.0f;
//const float g_pi = 3.1415926535897932384626433832795f;
const float g_defaultEpsilon = 0.000001f;

inline float saturate(float v) 
{
  return min(1.0f, max(0.0f, v));
}

#endif // __CUDACC__

#include "BaseMath.h"


CUDA_HOST_DEVICE_PREAMBLE float4 max(Float4ArgumentType a, Float4ArgumentType b)
{
  return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

CUDA_HOST_DEVICE_PREAMBLE float4 min(Float4ArgumentType a, Float4ArgumentType b)
{
  return make_float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

CUDA_HOST_DEVICE_PREAMBLE float3 max(Float3ArgumentType a, Float3ArgumentType b)
{
  return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

CUDA_HOST_DEVICE_PREAMBLE float3 min(Float3ArgumentType a, Float3ArgumentType b)
{
  return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

CUDA_HOST_DEVICE_PREAMBLE float2 max(Float2ArgumentType a, Float2ArgumentType b)
{
  return make_float2(max(a.x, b.x), max(a.y, b.y));
}

CUDA_HOST_DEVICE_PREAMBLE float2 min(Float2ArgumentType a, Float2ArgumentType b)
{
  return make_float2(min(a.x, b.x), min(a.y, b.y));
}

CUDA_HOST_DEVICE_PREAMBLE uint2 max(uint2 a, uint2 b)
{
  return make_uint2(max(a.x, b.x), max(a.y, b.y));
}

CUDA_HOST_DEVICE_PREAMBLE uint2 min(uint2 a, uint2 b)
{
  return make_uint2(min(a.x, b.x), min(a.y, b.y));
}

CUDA_HOST_DEVICE_PREAMBLE bool operator==(uint2 a, uint2 b)
{
  return a.x == b.x  && a.y == b.y;
}

CUDA_HOST_DEVICE_PREAMBLE bool operator!=(uint2 a, uint2 b)
{
  return a.x != b.x || a.y != b.y;
}

CUDA_DEVICE_PREAMBLE float2 saturate(Float2ArgumentType a)
{
  return make_float2(saturate(a.x), saturate(a.y));
}

CUDA_DEVICE_PREAMBLE float3 saturate(Float3ArgumentType a)
{
  return make_float3(saturate(a.x), saturate(a.y), saturate(a.z));
}

CUDA_DEVICE_PREAMBLE float4 saturate(Float4ArgumentType a)
{
  return make_float4(saturate(a.x), saturate(a.y), saturate(a.z), saturate(a.w));
}


CUDA_HOST_DEVICE_PREAMBLE void operator += (float2& a, Float2ArgumentType b)
{
  a.x += b.x;
  a.y += b.y;
}


CUDA_HOST_DEVICE_PREAMBLE void operator += (float3& a, Float3ArgumentType b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

CUDA_HOST_DEVICE_PREAMBLE void operator += (float4& a, Float3ArgumentType b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}


CUDA_HOST_DEVICE_PREAMBLE void operator -= (float3& a, Float3ArgumentType b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}


template <typename S, typename T>
CUDA_HOST_DEVICE_PREAMBLE float dot3(S a, T b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T>
CUDA_HOST_DEVICE_PREAMBLE float length3(T a)
{
  return sqrtf(dot3(a,a));
}


template <typename T>
CUDA_HOST_DEVICE_PREAMBLE float rlength3(T a)
{
#ifdef __CUDACC__
  return rsqrtf(dot3(a,a));
#else // !__CUDACC__
  return 1.0f / sqrtf(dot3(a,a));
#endif // __CUDACC__
}


template <typename T>
CUDA_HOST_DEVICE_PREAMBLE float rlength2(T a)
{
#ifdef __CUDACC__
  return rsqrtf(dot2(a,a));
#else // !__CUDACC__
  return 1.0f / sqrtf(dot2(a,a));
#endif // __CUDACC__
}


template <typename S, typename T>
CUDA_HOST_DEVICE_PREAMBLE float dot4(S a, T b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}



template <typename T>
CUDA_HOST_DEVICE_PREAMBLE float length4(T a)
{
  return sqrtf(dot4(a,a));
}

template <typename S, typename T>
CUDA_HOST_DEVICE_PREAMBLE float dot2(S a, T b)
{
  return a.x * b.x + a.y * b.y;
}

template <typename T>
CUDA_HOST_DEVICE_PREAMBLE float length2(T a)
{
  return sqrtf(dot2(a,a));
}

CUDA_HOST_DEVICE_PREAMBLE float2 normalize2(Float2ArgumentType a)
{
  float invL = rlength2(a);
  float2 r = {a.x * invL, a.y * invL};
  return r;
}


template <typename T>
CUDA_HOST_DEVICE_PREAMBLE float lengthSquared3(T a)
{
  return dot3(a,a);
}

template <typename T>
CUDA_HOST_DEVICE_PREAMBLE float lengthSquared2(T a)
{
  return dot2(a,a);
}


CUDA_HOST_DEVICE_PREAMBLE float4 operator + (Float4ArgumentType a, Float4ArgumentType b)
{
  float4 r = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
  return r;
}

CUDA_HOST_DEVICE_PREAMBLE float4 operator - (Float4ArgumentType a, Float4ArgumentType b)
{
  float4 r = {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
  return r;
}

CUDA_HOST_DEVICE_PREAMBLE float4 operator * (Float4ArgumentType a, float b)
{
  float4 r = {a.x * b, a.y * b, a.z * b, a.w * b};
  return r;
}

CUDA_HOST_DEVICE_PREAMBLE float4 normalize3(Float4ArgumentType a)
{
  float invL = rlength3(a);
  float4 r = {a.x * invL, a.y * invL, a.z * invL, a.w};
  return r;
}

CUDA_HOST_DEVICE_PREAMBLE float4 floor(Float4ArgumentType a)
{
  float4 r = { floorf(a.x), floorf(a.y), floorf(a.z), floorf(a.w) };
  return r;
}

CUDA_HOST_DEVICE_PREAMBLE float4 cross3(Float4ArgumentType a, Float4ArgumentType b)
{
  return make_float4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f);
}


CUDA_HOST_DEVICE_PREAMBLE float4 cross4(Float4ArgumentType a, Float4ArgumentType b, Float4ArgumentType c)
{
  return make_float4(c.w*b.y*a.z - b.w*c.y*a.z - c.w*a.y*b.z + a.w*c.y*b.z + b.w*a.y*c.z - a.w*b.y*c.z, 
					-c.w*b.x*a.z + b.w*c.x*a.z + c.w*a.x*b.z - a.w*c.x*b.z - b.w*a.x*c.z + a.w*b.x*c.z, 
					 c.w*b.x*a.y - b.w*c.x*a.y - c.w*a.x*b.y + a.w*c.x*b.y + b.w*a.x*c.y - a.w*b.x*c.y, 
					-c.x*b.y*a.z + b.x*c.y*a.z + c.x*a.y*b.z - a.x*c.y*b.z - b.x*a.y*c.z + a.x*b.y*c.z);
}


CUDA_HOST_DEVICE_PREAMBLE float clamp(float value, float mi, float ma)
{
  return min(max(value, mi), ma);
}

CUDA_HOST_DEVICE_PREAMBLE float3 operator + (Float3ArgumentType a, Float3ArgumentType b)
{
  float3 r = {a.x + b.x, a.y + b.y, a.z + b.z};
  return r;
}

CUDA_HOST_DEVICE_PREAMBLE float3 operator + (Float3ArgumentType a, float s)
{
  float3 r = {a.x + s, a.y + s, a.z + s};
  return r;
}


CUDA_HOST_DEVICE_PREAMBLE float3 operator - (Float3ArgumentType a, Float3ArgumentType b)
{
  float3 r = {a.x - b.x, a.y - b.y, a.z - b.z};
  return r;
}

CUDA_HOST_DEVICE_PREAMBLE float3 operator - (Float3ArgumentType a, float s)
{
  float3 r = {a.x - s, a.y - s, a.z - s};
  return r;
}



CUDA_HOST_DEVICE_PREAMBLE float3 operator - (Float3ArgumentType v)
{
  return make_float3(-v.x, -v.y, -v.z);
}

CUDA_HOST_DEVICE_PREAMBLE float4 operator - (Float4ArgumentType v)
{
  return make_float4(-v.x, -v.y, -v.z, -v.w);
}


CUDA_HOST_DEVICE_PREAMBLE float3 operator * (Float3ArgumentType a, Float3ArgumentType b)
{
  float3 r = {a.x * b.x, a.y * b.y, a.z * b.z};
  return r;
}

CUDA_HOST_DEVICE_PREAMBLE float3 operator * (Float3ArgumentType a, float b)
{
  float3 r = {a.x * b, a.y * b, a.z * b};
  return r;
}

CUDA_HOST_DEVICE_PREAMBLE float3 operator * (float s, Float3ArgumentType v)
{
  float3 r = {v.x * s, v.y * s, v.z * s};
  return r;
}


CUDA_HOST_DEVICE_PREAMBLE float3 operator / (Float3ArgumentType a, float b)
{
  float recip = 1.0f / b;
  return a * recip;
}


CUDA_HOST_DEVICE_PREAMBLE float3 operator / (Float3ArgumentType a, Float3ArgumentType b)
{
  float3 r = {a.x / b.x, a.y / b.y, a.z / b.z};
  return r;
}


CUDA_HOST_DEVICE_PREAMBLE void operator *= (float3 &a, float b)
{
  a.x *= b;
  a.y *= b;
  a.z *= b;
}


CUDA_HOST_DEVICE_PREAMBLE void operator *= (float4 &a, float b)
{
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}


CUDA_HOST_DEVICE_PREAMBLE void operator += (float4 &a, float b)
{
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}


CUDA_HOST_DEVICE_PREAMBLE void operator *= (float3 &a, Float3ArgumentType b)
{
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}


CUDA_HOST_DEVICE_PREAMBLE void operator *= (float4 &a, Float4ArgumentType b)
{
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}


CUDA_HOST_DEVICE_PREAMBLE float3 normalize3(Float3ArgumentType a)
{
  float invL = rlength3(a);
  float3 r = {a.x * invL, a.y * invL, a.z * invL};
  return r;
}

CUDA_HOST_DEVICE_PREAMBLE float3 floor(Float3ArgumentType a)
{
  float3 r = { floorf(a.x), floorf(a.y), floorf(a.z) };
  return r;
}

CUDA_HOST_DEVICE_PREAMBLE float3 cross3(Float3ArgumentType a, Float3ArgumentType b)
{
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}


CUDA_HOST_DEVICE_PREAMBLE float3 reflect3(Float3ArgumentType dir, Float3ArgumentType normal)
{
  return dir - normal * 2.0f * dot3(normal, dir);
}

CUDA_HOST_DEVICE_PREAMBLE float3 perpendicular3(Float3ArgumentType v)
{
  if (fabsf(v.x) < fabsf(v.y))
  {
      return make_float3(0.0f, -v.z, v.y);
  }

  return make_float3(-v.z, 0.0f, v.x);
}

CUDA_HOST_DEVICE_PREAMBLE float4 perpendicular3(Float4ArgumentType v)
{
  if (fabsf(v.x) < fabsf(v.y))
  {
      return make_float4(0.0f, -v.z, v.y, v.w);
  }

  return make_float4(-v.z, 0.0f, v.x, v.w);
}


CUDA_HOST_DEVICE_PREAMBLE float2 operator / (Float2ArgumentType a, float b)
{
  const float d = 1.0f / b;
  return make_float2(a.x * d, a.y * d);
}


CUDA_HOST_DEVICE_PREAMBLE float2 operator / (Float2ArgumentType a, Float2ArgumentType b)
{
  return make_float2(a.x / b.x, a.y / b.y);
}

CUDA_HOST_DEVICE_PREAMBLE float2 operator / (float a, Float2ArgumentType b)
{
  return make_float2(a / b.x, a / b.y);
}

CUDA_HOST_DEVICE_PREAMBLE float2 operator * (Float2ArgumentType a, float b)
{
  return make_float2(a.x * b, a.y * b);
}

CUDA_HOST_DEVICE_PREAMBLE float2 operator * (Float2ArgumentType a, Float2ArgumentType b)
{
  return make_float2(a.x * b.x, a.y * b.y);
}

CUDA_HOST_DEVICE_PREAMBLE float2 operator + (Float2ArgumentType a, float s)
{
  return make_float2(a.x + s, a.y + s);
}

CUDA_HOST_DEVICE_PREAMBLE float2 operator - (Float2ArgumentType a, float s)
{
  return make_float2(a.x - s, a.y - s);
}

CUDA_HOST_DEVICE_PREAMBLE float2 operator + (Float2ArgumentType a, Float2ArgumentType b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}

CUDA_HOST_DEVICE_PREAMBLE float2 operator - (Float2ArgumentType a, Float2ArgumentType b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}


CUDA_HOST_DEVICE_PREAMBLE uint2 operator * (uint2 a, uint32_t s)
{
  return make_uint2(a.x * s, a.y * s);
}

CUDA_HOST_DEVICE_PREAMBLE uint2 operator * (uint2 a, uint2 b)
{
  return make_uint2(a.x * b.x, a.y * b.y);
}


CUDA_HOST_DEVICE_PREAMBLE uint2 operator + (uint2 a, uint32_t s)
{
  return make_uint2(a.x + s, a.y + s);
}

CUDA_HOST_DEVICE_PREAMBLE uint2 operator - (uint2 a, uint32_t s)
{
  return make_uint2(a.x - s, a.y - s);
}


CUDA_HOST_DEVICE_PREAMBLE uint2 operator + (uint2 a, uint2 b)
{
  return make_uint2(a.x + b.x, a.y + b.y);
}

CUDA_HOST_DEVICE_PREAMBLE uint2 operator / (uint2 a, uint2 b)
{
  return make_uint2(a.x / b.x, a.y / b.y);
}

CUDA_HOST_DEVICE_PREAMBLE float4 make_float4(Float3ArgumentType a, float w = 0.0f)
{
  return make_float4(a.x, a.y, a.z, w);
}

CUDA_HOST_DEVICE_PREAMBLE float4 make_float4(Float2ArgumentType a, Float2ArgumentType b)
{
  return make_float4(a.x, a.y, b.x, b.y);
}

CUDA_HOST_DEVICE_PREAMBLE float3 make_float3(Float4ArgumentType a)
{
  return make_float3(a.x, a.y, a.z);
}

CUDA_HOST_DEVICE_PREAMBLE float3 make_float3(float a[3])
{
  return make_float3(a[0], a[1], a[2]);
}

CUDA_HOST_DEVICE_PREAMBLE float3 make_float3(float s)
{
  return make_float3(s, s, s);
}

CUDA_HOST_DEVICE_PREAMBLE float3 make_float3(Float2ArgumentType xy, float z)
{
  return make_float3(xy.x, xy.y, z);
}

CUDA_HOST_DEVICE_PREAMBLE float2 make_float2(uint2 a)
{
  return make_float2(float(a.x), float(a.y));
}

CUDA_HOST_DEVICE_PREAMBLE uint2 make_uint2(uint32_t a)
{
  return make_uint2(a, a);
}

CUDA_HOST_DEVICE_PREAMBLE uint2 make_uint2(uint3 a)
{
  return make_uint2(a.x, a.y);
}

CUDA_HOST_DEVICE_PREAMBLE uint2 clamp(uint2 value, uint2 mi, uint2 ma)
{
  return min(max(value, mi), ma);
}

CUDA_HOST_DEVICE_PREAMBLE float4 clamp(float4 value, float4 mi, float4 ma)
{
  return min(max(value, mi), ma);
}

CUDA_HOST_DEVICE_PREAMBLE int2 operator * (int2 a, int s)
{
  return make_int2(a.x * s, a.y * s);
}

CUDA_HOST_DEVICE_PREAMBLE int2 operator + (int2 a, int2 b)
{
  return make_int2(a.x + b.x, a.y + b.y);
}



template <typename T>
CUDA_HOST_DEVICE_PREAMBLE T interpolateVertexAttributes(const T *a, float u, float v)
{
  return a[0] * (1.0f - u - v) + a[1] * u + a[2] * v;
}

template <typename T, typename U>
CUDA_HOST_DEVICE_PREAMBLE T lerp(const T &a, const T &b, const U &t)
{
  return a + (b - a) * t;
}

#if 0
template <typename T>
CUDA_HOST_DEVICE_PREAMBLE void swap(T &a, T &b)
{
  T tmp(a);
  a = b;
  b = tmp;
}
#endif

/*
template <typename T>
CUDA_HOST_DEVICE_PREAMBLE T square(T a)
{
  return a * a;
}*/

CUDA_DEVICE_PREAMBLE uint32 ola_ilogbf(float a)
{
#if defined(__GNUC__)
	return ilogbf(a);
#else
#	ifdef __CUDACC__
	uint32 i = __float_as_int(a);
#	else // __CUDACC__
	float f = a;
	uint32 i = *(uint32_t*)&f;
#	endif // __CUDACC__
	uint32 expo = ((uint32)((i >> 23) & 0xff)) - 127;
	return expo;
#endif // ! __GNUC__
}

CUDA_DEVICE_PREAMBLE uint32 floorLog2_2(uint32 n)
{
  return (uint32)ola_ilogbf((float)n);
}

CUDA_HOST_DEVICE_PREAMBLE uint32 floorLog2(uint32 n)
{
#ifdef __CUDACC__
  return (int)ilogbf((float)n);
#elif defined(__GNUC__)
  return (int)logbf((float)n);
#else // __CUDACC__
  return (int)_logb((float)n);
#endif // __CUDACC__
#if 0
  int pos = 0;
  if (n >= 1<<16) { n >>= 16; pos += 16; }
  if (n >= 1<< 8) { n >>=  8; pos +=  8; }
  if (n >= 1<< 4) { n >>=  4; pos +=  4; }
  if (n >= 1<< 2) { n >>=  2; pos +=  2; }
  if (n >= 1<< 1) {           pos +=  1; }
  return ((n == 0) ? (-1) : pos);
#endif
}


inline bool epsilonEqual(float a, float b, float epsilon = g_defaultEpsilon)
{
  return fabsf(a - b) < epsilon;
}


enum DominantAxisIndex
{
  DAI_PX,
  DAI_PY,
  DAI_PZ,
  DAI_NX,
  DAI_NY,
  DAI_NZ,
  DAI_Max,
};

CUDA_HOST_DEVICE_PREAMBLE int clamp(int value, int mi, int ma)
{
  return min(max(value, mi), ma);
}

/**
 * Returns the number of blocks required to run 'numItems' threads, when
 * eahc block contains 'threadsPerBlock' threads. Note that there may be
 * at most threadsPerBlock - 1 extra threads that will try to index beyond
 * numItems.
 */
inline uint32_t getBlockCount(const uint32_t threadsPerBlock, const uint32_t numItems)
{
  return (numItems + threadsPerBlock - 1) / threadsPerBlock;
}
inline dim3 getBlockCount(const dim3 &bs, const uint2 &numItems)
{
  uint2 r = (numItems + make_uint2(bs.x, bs.y) - 1) / make_uint2(bs.x, bs.y);
  return dim3(r.x, r.y);
}
/**
 * Returns the nearest higher number to 'count' that is a multiple of 'multipleOf'
 */
inline uint32_t getNearestHigherMultiple(const uint32_t count, const uint32_t multipleOf)
{
  return getBlockCount(multipleOf, count) * multipleOf;
}

CUDA_HOST_DEVICE_PREAMBLE float3 powf(const float3 v, float e)
{
  return make_float3(powf(v.x, e), powf(v.y, e), powf(v.z, e));
}

CUDA_HOST_DEVICE_PREAMBLE float3 toSrgb(const float3 c)
{
  return powf(c, 1.0f / 2.2f);
}

CUDA_HOST_DEVICE_PREAMBLE float3 fromSrgb(const float3 c)
{
  return powf(c, 2.2f);
}

#endif // _CudaMath_h_
