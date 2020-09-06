/****************************************************************************/
/*!	\brief [chag::pp] Sort
 *
 * A stable radix sort based on the split primitives.
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

#ifndef _CHAG_PP_SORT_CUH_
#define _CHAG_PP_SORT_CUH_

//--//////////////////////////////////////////////////////////////////////////
//--	Include				///{{{1///////////////////////////////////////////

#include "ext/common.cuh"
#include "ext/range.cuh"
#include "ext/setup.cuh"

#include "detail/nta.hpp"
#include "detail/buffer.cuh"

#include "aspect/predicates.cuh"

CHAG_PP_ENTER_NAMESPACE()
//--	Functions			///{{{1///////////////////////////////////////////

/*! \brief Sort elements in range [aStart ... aEnd)
 *
 * Note: sort() needs two buffers to ping-pong in-between. The buffer holding
 * the final result will be returned.
 */
template< typename T > 
inline T* sort(
	const T* aStart, 
	const T* aEnd,
	T* aPing, T* aPong
);

template< 
	unsigned Low, 
	unsigned High, 
	template <unsigned,typename> class Pred,
	typename T
> 
inline T* sort(
	const T* aStart, 
	const T* aEnd,
	T* aPing, T* aPong
);

//--	Classes				///{{{1///////////////////////////////////////////
namespace aspect
{
	struct sort;
};

NTA_TEMPLATE(aspect::sort) struct Sorter
{
	NTA_DEFINE_CLASS_ASPECT(Aspect);

	typedef typename Aspect::input_type input_type;
	typedef typename Aspect::input_type output_type;

	static inline output_type* sort(
		const input_type* aStart, 
		const input_type* aEnd,
		output_type* aPing, 
		output_type* aPong, 
		SizeType* aCountBuffer = 0,
		SizeType* aOffsetBuffer = 0
	);

	public: /* private! nvcc messes this up when creating the .foo.cpp */
		typedef typename Aspect::kern_setup KSetup;
		typedef typename Aspect::template job_setup<
			input_type, KSetup, detail::HasPair<input_type>::value
		> JSetup;
		typedef typename JSetup::KernelParam KParam;

		enum { LIMIT_LOW = Aspect::LIMIT_LOW };
		enum { LIMIT_HIGH = Aspect::LIMIT_HIGH };

		template< unsigned P > struct Pred 
			: public Aspect::template predicate<P,input_type>
		{};

		enum { STAGE = 1 };
};

namespace aspect
{
	NTA_ASPECT(sort)
	{
		NTA_DECLARE_ARGUMENT_CONST( LimitLow, LIMIT_LOW, 0 );
		NTA_DECLARE_ARGUMENT_CONST( LimitHigh, LIMIT_HIGH, 32 );

		NTA_DECLARE_ARGUMENT_TYPENAME( InputType, input_type, unsigned );
		NTA_DECLARE_ARGUMENT_TYPENAME( KSetup, kern_setup, KernelSetupWarp<> );

		NTA_DECLARE_ARGUMENT_TEMPLATE( Predicate, predicate, pred::BitUnset,
			(unsigned P, typename T), (P,T)
		);
		NTA_DECLARE_ARGUMENT_TEMPLATE( JobSetup, job_setup, JobSetupStatic,
			(typename T, class KSetup, bool UsePairs), (T, KSetup, UsePairs)
		);
	};
};

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#include "detail/sort.cu.inl"
#endif // _CHAG_PP_SORT_CUH_
