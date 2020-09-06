/****************************************************************************/
/*!	\brief [chag::pp] Split
 *
 * Split: a.k.a. partition.
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

#ifndef _CHAG_PP_SPLIT_CUH_
#define _CHAG_PP_SPLIT_CUH_

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

/*! \brief Compact elements in range
 *
 * Compact elements in range \left[\a aStart ... \a aEnd\right]. 
 *
 * At most (\a aStart - \a aEnd) valid elements are stored at \a aOutput and 
 * following locations. The actual number of valid elements is stored at
 * \a aNumValid if it is non-null.
 */
template< typename T > 
inline void split(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	SizeType* aNumValid = 0
);
template< typename T, class Predicate > 
inline void split(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	SizeType* aNumValid,
	const Predicate& aPredicate
);

//--	Classes				///{{{1///////////////////////////////////////////
namespace aspect
{
	struct split;
};

NTA_TEMPLATE(aspect::split) struct Splitter
{
	NTA_DEFINE_CLASS_ASPECT(Aspect);

	typedef typename Aspect::input_type input_type;
	typedef typename Aspect::input_type output_type;

	template< class Predicate >
	static inline void split(
		const input_type* aStart, 
		const input_type* aEnd,
		output_type* aOutput, 
		SizeType* aNumValid,
		const Predicate& aPredicate,
		SizeType* aCountBuffer = 0,
		SizeType* aOffsetBuffer = 0
	);

	public: /* private! nvcc messes this up when creating the .foo.cpp */
		typedef typename Aspect::kern_setup KSetup;
		typedef typename Aspect::template job_setup<
			input_type, KSetup, detail::HasPair<input_type>::value
		> JSetup;
		typedef typename JSetup::KernelParam KParam;

		enum { STAGE = Aspect::template alg_select<input_type>::STAGE };
};

namespace aspect
{
	template< typename T >
	struct SplitAlgSelector
	{
		enum { STAGE = detail::HasPair<T>::value };
	};

	NTA_ASPECT(split)
	{
		NTA_DECLARE_ARGUMENT_TYPENAME( InputType, input_type, unsigned );
		NTA_DECLARE_ARGUMENT_TYPENAME( KSetup, kern_setup, KernelSetupWarp<> );

		NTA_DECLARE_ARGUMENT_TEMPLATE( JobSetup, job_setup, JobSetupStatic,
			(typename T, class KSetup, bool UsePairs), (T, KSetup, UsePairs)
		);

		NTA_DECLARE_ARGUMENT_TEMPLATE( AlgSelect, alg_select, SplitAlgSelector,
			(typename T), (T)
		);
	};
};

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#include "detail/split.cu.inl"
#endif // _CHAG_PP_SPLIT_CUH_
