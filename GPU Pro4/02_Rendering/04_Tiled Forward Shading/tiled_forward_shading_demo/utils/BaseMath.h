#ifndef BASEMATH_H_D3BC8986_09BB_4572_B2E7_D291196AD04D
#define BASEMATH_H_D3BC8986_09BB_4572_B2E7_D291196AD04D

#if defined(BASEMATH_FUNC_PREAMBLE)
#	define _FUNC_PREAMBLE_ BASEMATH_FUNC_PREAMBLE
#else
#	define _FUNC_PREAMBLE_ inline
#endif // BASEMATH_FUNC_PREAMBLE

/* g_pi is ambiguous (defined in both CudaMath.h, Math.h. Other fixes failed,
 * so use this as an ugly workaround.
 */
const float g_pi = 3.1415926535897932384626433832795f;

/* Same problem with square()
 */
template< typename T > _FUNC_PREAMBLE_
const T square( T a )
{
	return a*a;
}

_FUNC_PREAMBLE_ float toRadians(float degrees)
{
	return degrees * g_pi / 180.0f;
}

template <typename T> 
_FUNC_PREAMBLE_ T sgn(T val)
{
    return T((val > T(0)) - (val < T(0)));
}


#undef _FUNC_PREAMBLE_

#endif // BASEMATH_H_D3BC8986_09BB_4572_B2E7_D291196AD04D
