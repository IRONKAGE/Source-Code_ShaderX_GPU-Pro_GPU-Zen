/****************************************************************************/
/*!	\brief [chag::pp] Reduction
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

#ifndef _CHAG_PP_REDUCE_CUH_
#define _CHAG_PP_REDUCE_CUH_

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

/*! \brief Reduce elements in range
 *
 * Reduce elements in range \left[\a aStart ... \a aEnd\right], using operator
 * \a aOperator, or op::Add<>() if not specified.
 *
 * The single output value is stored in \a aOutput[0].
 */
template< typename T > 
inline void reduce(
	const T* aStart, 
	const T* aEnd,
	T* aOutput
);
template< typename T, class Op > 
inline void reduce(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	const Op& aOperator
);

//--	Classes				///{{{1///////////////////////////////////////////
namespace aspect
{
	struct reduce;
};

NTA_TEMPLATE(aspect::reduce) struct Reducer
{
	NTA_DEFINE_CLASS_ASPECT(Aspect);

	typedef typename Aspect::input_type input_type;
	typedef typename Aspect::input_type output_type;

	template< class Op >
	static inline void reduce(
		const input_type* aStart, 
		const input_type* aEnd,
		output_type* aOutput, 
		const Op& aOperator,
		input_type* aReduceBuffer = 0
	);

	public: /* private! nvcc messes this up when creating the .foo.cpp */
		typedef typename Aspect::kern_setup KSetup;
		typedef typename Aspect::template job_setup<
			input_type, KSetup, detail::HasPair<input_type>::value
		> JSetup;
		typedef typename JSetup::KernelParam KParam;
};

namespace aspect
{
	NTA_ASPECT(reduce)
	{
		NTA_DECLARE_ARGUMENT_TYPENAME( InputType, input_type, unsigned );
		NTA_DECLARE_ARGUMENT_TYPENAME( KSetup, kern_setup, KernelSetupWarp<> );

		NTA_DECLARE_ARGUMENT_TEMPLATE( JobSetup, job_setup, JobSetupStatic,
			(typename T, class KSetup, bool UsePairs), (T, KSetup, UsePairs)
		);
	};
};

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#include "detail/reduce.cu.inl"
#endif // _CHAG_PP_REDUCE_CUH_
