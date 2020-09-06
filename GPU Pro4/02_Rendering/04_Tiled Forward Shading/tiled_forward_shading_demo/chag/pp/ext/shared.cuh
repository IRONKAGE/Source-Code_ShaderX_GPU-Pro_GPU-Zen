/****************************************************************************/
/*!	\brief [chag::pp] shared memory helpers
 *
 * Problem: shared memory needs to be 'volatile', otherwise the compiler will
 * perform bad optimizations and break our code.
 *
 * Using 'volatile' works well for most types, except the built-in vector
 * types, e.g. uint2 & co.
 * 
 * For uint2 & co. we will get errors like
 *		error: no operator "=" matches these operands
 *          operand types are: volatile uint2 = volatile uint2
 * 
 * Fun.
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

#ifndef _CHAG_PP_EXT_SHARED_CUH_
#define _CHAG_PP_EXT_SHARED_CUH_

//--//////////////////////////////////////////////////////////////////////////
CHAG_PP_ENTER_NAMESPACE()
//--	Include				///{{{1///////////////////////////////////////////

#include "traits.cuh"

//--	Volatile-fix		///{{{1///////////////////////////////////////////
namespace detail
{
	/*template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsPlain<T>::value, void) sm_set( 
		volatile T* aSm,
		unsigned aIndex,
		const T& aIn
	);
	template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsPlain<T>::value, T) sm_get( 
		volatile T* aSm,
		unsigned aIndex
	);

	template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsPair<T>::value, void) sm_set( 
		volatile T* aSm,
		unsigned aIndex,
		const T& aIn
	);
	template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsPair<T>::value, T) sm_get( 
		volatile T* aSm,
		unsigned aIndex
	);

	template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsQuad<T>::value, void) sm_set( 
		volatile T* aSm,
		unsigned aIndex,
		const T& aIn
	);
	template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsQuad<T>::value, T) sm_get( 
		volatile T* aSm,
		unsigned aIndex
	);*/
}

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#include "detail/shared.cu.inl"
#endif // _CHAG_PP_EXT_SHARED_CUH_
