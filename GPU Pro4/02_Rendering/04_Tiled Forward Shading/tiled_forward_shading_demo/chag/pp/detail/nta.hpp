/****************************************************************************/
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
/*!	\brief Named template arguments
 *
 * Helper macros for named template arguments. Example:
 * \code
 *	// Stuff
 *	template< typename T > struct A {};
 *	template< typename T > struct B {};
 *
 *  // Default argument values
 * 	struct ExampleDefaultPack
 * 	{
 * 		enum { CONST = 42 };
 *
 * 		typedef unsigned value_type;
 * 		typedef float result_type;
 *
 * 		template< typename T >
 * 		struct bar : public A<T> {};
 * 	};
 *
 *  // Argument Symbols
 * 	NTA_DECLARE_ARGUMENT_CONST( ExampleDefaultPack, ExConst, CONST );
 *
 * 	NTA_DECLARE_ARGUMENT_TYPENAME( ExampleDefaultPack, ValueType, value_type );
 * 	NTA_DECLARE_ARGUMENT_TYPENAME( ExampleDefaultPack, ResultType, result_type );
 *
 * 	NTA_DECLARE_ARGUMENT_TEMPLATE( ExampleDefaultPack, Bar, bar,
 * 		(typename T), (T)
 * 	);
 *
 *  // Class
 *  NTA_TEMPLATE(ExampleDefaultPack) class Example
 *  {
 *  	NTA_DEFINE_CLASS_ASPECT(Aspect);
 *	
 *		// "Usage"
 *		typedef Aspect::template bar<typename Aspect::value_type> bar_type;
 *  	typename Aspect::result_type foo( typename Aspect::value_type );
 *  };
 *
 *	// Usage
 *	Example< 
 *		ValueType<int>, Bar<B>
 *	> exampleInstance;
 *
 * \endcode
 */
/****************************************************************************/

#ifndef _NAMED_TEMPLATE_ARGUMENTS_INC_
#define _NAMED_TEMPLATE_ARGUMENTS_INC_

//--//////////////////////////////////////////////////////////////////////////
//--	Macros				///{{{1///////////////////////////////////////////

#define NTA_ASPECT(name)													\
	struct name : public _nta_detail::Base<name>

#define NTA_DECLARE_ARGUMENT_CONST( name, id, def )							\
	enum { id = def };														\
																			\
	template< unsigned _C > struct name 									\
		: virtual public _nta_self_const<_C>::self {						\
		enum { id = _C };													\
	}
#define NTA_DECLARE_ARGUMENT_TYPENAME( name, id, def )						\
	typedef def id;															\
																			\
	template< typename _T > struct name 									\
		: virtual public _nta_self_ty<_T>::self {							\
		typedef _T id;														\
	}

#define NTA_UNPACK_ARGS_(...) __VA_ARGS__
#define NTA_DECLARE_ARGUMENT_TEMPLATE( name, id, def, a, b ) 				\
	template< NTA_UNPACK_ARGS_ a > struct id 								\
		: public def< NTA_UNPACK_ARGS_ b > {};								\
																			\
	template< template< NTA_UNPACK_ARGS_ a > class _C >						\
	struct _nta_self_##name { typedef _nta_self self; };					\
																			\
	template< template< NTA_UNPACK_ARGS_ a > class _T > 					\
	struct name : virtual public _nta_self_##name<_T>::self {				\
		template< NTA_UNPACK_ARGS_ a > struct id 							\
			: public _T< NTA_UNPACK_ARGS_ b > {};							\
	}

#define NTA_TEMPLATE(aspect)											 	\
	template<																\
		class _Policy0 = aspect,											\
		class _Policy1 = aspect,											\
		class _Policy2 = aspect,											\
		class _Policy3 = aspect,											\
		class _Policy4 = aspect,											\
		class _Policy5 = aspect,											\
		class _Policy6 = aspect,											\
		class _Policy7 = aspect,											\
		class _Policy8 = aspect,											\
		class _Policy9 = aspect												\
	>

#define NTA_TEMPLATE_PREAMBLE												\
	template< class _Policy0, class _Policy1, class _Policy2, 				\
		class _Policy3, class _Policy4, class _Policy5, class _Policy6,		\
		class _Policy7, class _Policy8, class _Policy9						\
	>
#define NTA_TEMPLATE_ARGUMENTS												\
	_Policy0, _Policy1, _Policy2, _Policy3, _Policy4, _Policy5, _Policy6,	\
	_Policy7, _Policy8, _Policy9

#define NTA_DEFINE_CLASS_ASPECT(name)										\
	typedef ::_nta_detail::Selector<_Policy0,_Policy1,_Policy2,				\
		_Policy3,_Policy4,_Policy5,_Policy6,_Policy7,_Policy8,_Policy9		\
	> name		

//--	Classes				///{{{1///////////////////////////////////////////
namespace _nta_detail 
{
	class Empty {};

	template< class Aspect >
	struct Base
	{
		typedef Aspect _nta_self;

		template< typename T > struct _nta_self_ty { typedef Aspect self; };
		template< unsigned C > struct _nta_self_const { typedef Aspect self; };
	};

	template< class Base, int Tag >
	class Discriminator : virtual public Base {};


	template< class S0 = Empty, class S1 = Empty, class S2 = Empty,
		class S3 = Empty, class S4 = Empty, class S5 = Empty, class S6 = Empty,
		class S7 = Empty, class S8 = Empty, class S9 = Empty
	> class Selector : 
		public Discriminator<S0,0>,
		public Discriminator<S1,1>,
		public Discriminator<S2,2>,
		public Discriminator<S3,3>,
		public Discriminator<S4,4>,
		public Discriminator<S5,5>,
		public Discriminator<S6,6>,
		public Discriminator<S7,7>,
		public Discriminator<S8,8>,
		public Discriminator<S9,9>
	{};
};

//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#endif // _NAMED_TEMPLATE_ARGUMENTS_INC_
