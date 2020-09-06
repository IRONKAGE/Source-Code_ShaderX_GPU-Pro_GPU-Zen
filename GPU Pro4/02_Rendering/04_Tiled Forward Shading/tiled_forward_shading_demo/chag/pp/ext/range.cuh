/****************************************************************************/
/*!	\brief [chag::pp] Perform operations on ranges
 *
 * General notes:
 *		- In variants that use two shared memory buffers (for calculation of
 *			the prefix sum and staging of output), it is admissible to have 
 *			those two memory areas alias. 
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

#ifndef _CHAG_PP_EXT_RANGE_CUH_
#define _CHAG_PP_EXT_RANGE_CUH_

//--//////////////////////////////////////////////////////////////////////////
//--	Include				///{{{1///////////////////////////////////////////

#include "common.cuh"
#include "traits.cuh"

#include "unit.cuh"
#include "setup.cuh"

#include "../aspect/operators.cuh"

CHAG_PP_ENTER_NAMESPACE()
//--	Functions			///{{{1///////////////////////////////////////////
template< 
	typename T,
	class KSetup,
	bool UsePairs = detail::HasPair<T>::value
> struct Range
{
	// Reduction
	template< class Op >
	static _CHAG_PP_DEV T reduce( 
		const T* aStart, const T* aEnd,
		const Op& aOp,
		volatile T* aSmReduce
	);

	template< class Pred >
	static _CHAG_PP_DEV SizeType count(
		const T* aStart, const T* aEnd,
		const Pred& aPred,
		volatile SizeType* aSmReduce
	);
	
	// Prefix / Scan
	template< class Op >
	static _CHAG_PP_DEV T prefix(
		const T* aStart, const T* aEnd,
		T* aOut,
		T aOffset,
		const Op& aOp,
		volatile T* aSmPrefix
	);
	template< class Op >
	static _CHAG_PP_DEV T prefix_inclusive(
		const T* aStart, const T* aEnd,
		T* aOut,
		T aOffset,
		const Op& aOp,
		volatile T* aSmPrefix
	);

	// Compaction
	template< class Pred >
	static _CHAG_PP_DEV SizeType compact_scatter(
		const T* aStart, const T* aEnd,
		T* aOut,
		const Pred& aPred,
		volatile SizeType* aSmPrefix
	);
	template< class Pred >
	static _CHAG_PP_DEV SizeType compact_stage(
		const T* aStart, const T* aEnd,
		T* aOut,
		const Pred& aPred,
		volatile T* aSmStage,
		volatile SizeType* aSmPrefix
	);

	// Split
	template< class Pred >
	static _CHAG_PP_DEV SizeType split_scatter(
		const T* aStart, const T* aEnd,
		T* aLeft, T* aRight,
		const Pred& aPred,
		volatile SizeType* aSmPrefix
	);
	template< class Pred >
	static _CHAG_PP_DEV SizeType split_stage(
		const T* aStart, const T* aEnd,
		T* aLeft, T* aRight,
		const Pred& aPred,
		volatile T* aSmStage,
		volatile SizeType* aSmPrefix
	);

	// Special: 4-way split & "4-way count"
	typedef uint4 SizeType4;

	template< class Pred0, class Pred1 >
	static _CHAG_PP_DEV SizeType4 count4(
		const T* aStart, const T* aEnd,
		const Pred0& aPred0,
		const Pred1& aPred1,
		volatile SizeType* aSmReduce 
	);
	template< class Pred0, class Pred1 >
	static _CHAG_PP_DEV SizeType4 count4(
		const T* aStart, const T* aEnd,
		const Pred0& aPred0,
		const Pred1& aPred1,
		volatile SizeType* aSmCount,  // 4xSIMD elements!
		volatile SizeType* aSmReduce 
	);


	template< class Pred0, class Pred1 >
	static _CHAG_PP_DEV void split4_scatter(
		const T* aStart, const T* aEnd,
		T* aBase, volatile SizeType* aOffsets,
		const Pred0& aPred0,
		const Pred1& aPred1,
		volatile SizeType* aSmPrefix
	);

	/* TODO
	template< class Pred0, class Pred1 >
	static _CHAG_PP_DEV void split4_stage(
		const T* aStart, const T* aEnd,
		T* aBase, volatile SizeType* aOffsets,
		const Pred0& aPred0,
		const Pred1& aPred1,
		volatile T* aSmStage,
		volatile SizeType* aSmPrefix
	);*/

	protected:
		typedef typename KSetup::Env _Env;
		typedef Unit<KSetup> _Unit;

	/* Shared memory requirements, in number of elements 
	 */
	public:
		enum { MEMORY_ELEMENTS_STAGE = _Env::SIMD * (UsePairs ? 2:1) };

		enum { MEMORY_ELEMENTS_REDUCE = _Unit::MEMORY_ELEMENTS_REDUCE };
		enum { MEMORY_ELEMENTS_PREFIX = _Unit::MEMORY_ELEMENTS_PREFIX };
};

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#include "detail/range.cu.inl"
#endif // _CHAG_PP_EXT_RANGE_CUH_
