/****************************************************************************/
/*!	\brief [chag::pp] Operators
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

#ifndef _CHAG_PP_ASPECT_OPERATORS_CUH_
#define _CHAG_PP_ASPECT_OPERATORS_CUH_

//--//////////////////////////////////////////////////////////////////////////
//--	Include				///{{{1///////////////////////////////////////////

#include "../ext/common.cuh"

CHAG_PP_ENTER_NAMESPACE()
//--	Classes				///{{{1///////////////////////////////////////////
namespace op
{
	template< typename T >
	struct Add
	{
		_CHAG_PP_ALL T operator() ( T aA, T aB ) const
		{
			return aA + aB;
		}

		_CHAG_PP_ALL T identity() const 
		{
			return T(0);
		}
	};
	template< typename T >
	struct Sub
	{
		_CHAG_PP_ALL T operator() ( T aA, T aB ) const
		{
			return aA - aB;
		}

		_CHAG_PP_ALL T identity() const 
		{
			return T(0);
		}
	};

	template<> struct Add< ::float3 >
	{
		_CHAG_PP_ALL ::float3 operator() ( ::float3 aA, ::float3 aB ) const
		{
			return make_float3( aA.x+aB.x, aA.y+aB.y, aA.z+aB.z );
		}

		_CHAG_PP_ALL ::float3 identity() const 
		{
			return make_float3( 0.0f, 0.0f, 0.0f );
		}
	};
}

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#endif // _CHAG_PP_ASPECT_OPERATORS_CUH_
