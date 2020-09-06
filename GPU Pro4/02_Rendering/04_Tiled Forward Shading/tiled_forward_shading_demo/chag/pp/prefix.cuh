/****************************************************************************/
/*!	\brief [chag::pp] Prefix
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

#ifndef _CHAG_PP_PREFIX_CUH_
#define _CHAG_PP_PREFIX_CUH_

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

/*! \brief Scan elements in range
 *
 * Scan elements in range \left[\a aStart ... \a aEnd\right], using the 
 * optional operator \a aOperator. The default operator is op::Add<>.
 *
 * This is a exclusive prefix operation.
 */
template< typename T > 
inline void prefix(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	T* aTotal = 0
);
template< typename T, class Op > 
inline void prefix(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	T* aTotal,
	const Op& aOperator
);

template< typename T > 
inline void prefix_inclusive(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	T* aTotal = 0
);
template< typename T, class Op > 
inline void prefix_inclusive(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	T* aTotal,
	const Op& aOperator
);


//--	Classes				///{{{1///////////////////////////////////////////
namespace aspect
{
	struct prefix;
};

NTA_TEMPLATE(aspect::prefix) struct Prefixer
{
	NTA_DEFINE_CLASS_ASPECT(Aspect);

	typedef typename Aspect::input_type input_type;
	typedef typename Aspect::input_type output_type;

	template< class Op >
	static inline void prefix(
		const input_type* aStart, 
		const input_type* aEnd,
		output_type* aOutput, 
		output_type* aTotal,
		const Op& aOperator,
		output_type* aPartialBuffer = 0,
		output_type* aPartialOffsets = 0
	);

	public: /* private! nvcc messes this up when creating the .foo.cpp */
		enum { INCLUSIVE = Aspect::inclusive_scan };

		typedef typename Aspect::kern_setup KSetup;
		typedef typename Aspect::template job_setup<
			input_type, KSetup, detail::HasPair<input_type>::value
		> JSetup;
		typedef typename JSetup::KernelParam KParam;
};

namespace aspect
{
	NTA_ASPECT(prefix)
	{
		NTA_DECLARE_ARGUMENT_TYPENAME( InputType, input_type, unsigned );
		NTA_DECLARE_ARGUMENT_TYPENAME( KSetup, kern_setup, KernelSetupWarp<> );

		NTA_DECLARE_ARGUMENT_TEMPLATE( JobSetup, job_setup, JobSetupStatic,
			(typename T, class KSetup, bool UsePairs), (T, KSetup, UsePairs)
		);

		NTA_DECLARE_ARGUMENT_CONST( Inclusive, inclusive_scan, false );
	};
};

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#include "detail/prefix.cu.inl"
#endif // _CHAG_PP_PREFIX_CUH_
