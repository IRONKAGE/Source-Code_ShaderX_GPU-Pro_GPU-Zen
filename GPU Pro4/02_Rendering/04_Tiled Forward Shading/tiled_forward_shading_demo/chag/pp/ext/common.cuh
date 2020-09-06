/****************************************************************************/
/*!	\brief [chag::pp] common header
 */
/* Copyright (c) 2009, Markus Billeter, Ola Olsson and Ulf Assarsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/****************************************************************************/

#ifndef _CHAG_PP_EXT_COMMON_CUH_
#define _CHAG_PP_EXT_COMMON_CUH_

//--//////////////////////////////////////////////////////////////////////////
//--	Algorithm			///{{{1///////////////////////////////////////////

#include <cstddef>

//--	Macros				///{{{1///////////////////////////////////////////

#define CHAG_PP_ENTER_NAMESPACE() namespace chag { namespace pp {
#define CHAG_PP_LEAVE_NAMESPACE() } }

//--	Defines				///{{{1///////////////////////////////////////////

#define _CHAG_PP_FORCEINLINE __forceinline__

#define _CHAG_PP_KRN __global__
#define _CHAG_PP_DEV __device__ _CHAG_PP_FORCEINLINE
#define _CHAG_PP_ALL __device__ __host__ _CHAG_PP_FORCEINLINE


#define CHAG_PP_DEFAULT_BLOCKS 120
#define CHAG_PP_DEFAULT_THREADS 128

CHAG_PP_ENTER_NAMESPACE()
//--	Types				///{{{1///////////////////////////////////////////

typedef unsigned SizeType;

//--	Helpers				///{{{1///////////////////////////////////////////
namespace detail
{
	template< unsigned N, unsigned SIMD >
	_CHAG_PP_DEV void conditional_barrier();

	template< unsigned N > 
	struct NextPowerOf2;
};

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#include "detail/common.cu.inl"
#endif // _CHAG_PP_EXT_COMMON_CUH_
