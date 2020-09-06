/****************************************************************************/
/*!	\brief [chag::pp] type traits
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

#ifndef _CHAG_PP_EXT_TRAITS_CUH_
#define _CHAG_PP_EXT_TRAITS_CUH_

//--//////////////////////////////////////////////////////////////////////////
CHAG_PP_ENTER_NAMESPACE()
//--	Weird NVCC types	///{{{1///////////////////////////////////////////
namespace detail
{
	template< typename T > struct IsPair
	{
		enum { value = 0 };
	};

#define _chag_pp_is_pair(ty)										\
	template<> struct IsPair<ty> { enum { value = 1 }; };			\
	/*END*/

	_chag_pp_is_pair(::int2);
	_chag_pp_is_pair(::uint2);
	_chag_pp_is_pair(::float2);
#undef _chag_pp_is_pair

	template< typename T > struct IsTri
	{
		enum { value = 0 };
	};

#define _chag_pp_is_tri(ty)										\
	template<> struct IsTri<ty> { enum { value = 1 }; };			\
	/*END*/

	_chag_pp_is_tri(::int3);
	_chag_pp_is_tri(::uint3);
	_chag_pp_is_tri(::float3);
#undef _chag_pp_is_tri

	template< typename T > struct IsQuad
	{
		enum { value = 0 };
	};

#define _chag_pp_is_quad(ty)										\
	template<> struct IsQuad<ty> { enum { value = 1 }; };			\
	/*END*/

	_chag_pp_is_quad(::int4);
	_chag_pp_is_quad(::uint4);
	_chag_pp_is_quad(::float4);
#undef _chag_pp_is_quad


	template< typename T > struct IsPlain
	{
		enum { value = ((IsPair<T>::value || IsTri<T>::value || IsQuad<T>::value) ? 0 : 1) };
	};
}

//--	Type <-> Pairs		///{{{1///////////////////////////////////////////
namespace detail
{
	template< typename T > struct HasPair
	{
		enum { value = 0 };
	};

	template< typename T > struct PairTypeOf
	{
		typedef void type;
	};
	template< typename T > struct PrimTypeOf
	{
		typedef void type;
	};
	
#define _chag_pp_enable_pair(ty, pair) 								\
	template<> struct HasPair<ty> { enum { value = 1 }; };			\
	template<> struct PairTypeOf<ty> { typedef pair type; };		\
	template<> struct PrimTypeOf<pair> { typedef ty type; }			\
	/*END*/

	_chag_pp_enable_pair( int, ::int2 );
	_chag_pp_enable_pair( float, ::float2 );
	_chag_pp_enable_pair( unsigned, ::uint2 );

#undef _chag_pp_enable_pair
};

//--	Type <-> Tri		///{{{1///////////////////////////////////////////
namespace detail
{
#	define chag_pp_tri_(tri,ty) \
		template<> struct PrimTypeOf<tri> { typedef ty type; } \
	/*ENDM*/

	chag_pp_tri_( ::int3, int );
	chag_pp_tri_( ::uint3, unsigned );	
	chag_pp_tri_( ::float3, float );

#	undef chag_pp_tri_
}

//--	Type <-> Quad		///{{{1///////////////////////////////////////////
namespace detail
{
#	define chag_pp_quad_(quad,ty) \
		template<> struct PrimTypeOf<quad> { typedef ty type; } \
	/*ENDM*/

	chag_pp_quad_( ::int4, int );
	chag_pp_quad_( ::uint4, unsigned );	
	chag_pp_quad_( ::float4, float );

#	undef chag_pp_quad_
}

//--	Enable If			///{{{1///////////////////////////////////////////

/* Based on Boost.EnableIf. It's only a few LOC, so we'll avoid depending on
 * all of Boost just because of this.
 */

namespace detail
{
	template< bool, typename T = void >
	struct enable_if
	{};

	template< typename T >
	struct enable_if<true,T>
	{
		typedef T type;
	};
}

#define _CHAG_PP_IF(pred, ty) typename detail::enable_if<pred,ty>::type

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#endif // _CHAG_PP_EXT_TRAITS_CUH_
