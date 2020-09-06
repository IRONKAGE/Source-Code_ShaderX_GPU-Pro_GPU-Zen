/****************************************************************************/
/*!	\brief [chag::pp] Predicates
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

#ifndef _CHAG_PP_ASPECT_PREDICATES_CUH_
#define _CHAG_PP_ASPECT_PREDICATES_CUH_

//--//////////////////////////////////////////////////////////////////////////
//--	Include				///{{{1///////////////////////////////////////////

#include "../ext/common.cuh"

CHAG_PP_ENTER_NAMESPACE()
//--	Classes				///{{{1///////////////////////////////////////////
namespace pred
{
	template< typename T >
	struct Nonzero
	{
		_CHAG_PP_ALL bool operator () ( T aA ) const
		{
			return aA != 0;
		}
	};

	template< unsigned BIT, typename T >
	struct BitSet
	{
		_CHAG_PP_ALL bool operator () ( T aA ) const
		{
			return aA & (1u<<BIT);
		}
	};
	template< unsigned BIT, typename T >
	struct BitUnset
	{
		_CHAG_PP_ALL bool operator () ( T aA ) const
		{
			return !(aA & (1u<<BIT));
		}
	};
	
	template< typename T >
	struct Even
	{
		_CHAG_PP_ALL bool operator () ( T aA ) const
		{
			return aA % 2 == 0;
		}
	};
	template<> struct Even<uint2>
	{
		_CHAG_PP_ALL bool operator () ( uint2 aA ) const
		{
			return aA.x % 2 == 0;
		}
	};

	template< template<typename> class Pred, typename T >
	struct Inverse
	{
		Pred<T> pred;

		_CHAG_PP_ALL bool operator() ( T aA ) const
		{
			return pred(aA);
		}
	};
}

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#endif // _CHAG_PP_ASPECT_PREDICATES_CUH_
