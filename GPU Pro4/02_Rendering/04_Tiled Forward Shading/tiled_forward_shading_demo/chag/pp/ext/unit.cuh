/****************************************************************************/
/*!	\brief [chag::pp] Unit<> class
 *
 * Unit<> defined a number of methods that operate with on a whole unit. (A
 * unit can be a single thread, a warp or a whole block - any collection of
 * threads that can be synchronized independently of other collections.)
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

#ifndef _CHAG_PP_EXT_UNIT_CUH_
#define _CHAG_PP_EXT_UNIT_CUH_

//--//////////////////////////////////////////////////////////////////////////
//--	Include				///{{{1///////////////////////////////////////////

#include "common.cuh"
#include "setup.cuh"
#include "shared.cuh"

#include "../aspect/operators.cuh"

CHAG_PP_ENTER_NAMESPACE()
//--	Classes				///{{{1///////////////////////////////////////////
template< 
	class KSetup 
> struct Unit
{
	// Reduce
	template< typename T, class Op >
	static _CHAG_PP_DEV T reduce( 
		const T& aVal,
		const Op& aOp, 
		volatile T* aSmReduce
	);

	// Prefix / Scan
	template< typename T, class Op >
	static _CHAG_PP_DEV T prefix( 
		const T& aVal, 
		const Op& aOp, 
		volatile T* aSmPrefix
	);
	template< typename T, class Op >
	static _CHAG_PP_DEV T prefix_inclusive( 
		const T& aVal, 
		const Op& aOp, 
		volatile T* aSmPrefix
	);

	template< typename T >
	static _CHAG_PP_DEV T prefix_total_get( volatile T* aSmPrefix );

	// Segmented Prefix / Scan 
	template< typename T, class Op >
	static _CHAG_PP_DEV T segmented_prefix_inclusive(
		const T& aVal,
		const Op& aOp,
		bool aSegmentStart,
		volatile T* aSmPrefix,
		volatile unsigned* aSmSegments
	);
	template< typename T, class Op >
	static _CHAG_PP_DEV T segmented_prefix_inclusive(
		const T& aVal,
		const T& aStartOffset, // only needs to be valid in lane 0
		const Op& aOp,
		bool aSegmentStart,
		volatile T* aSmPrefix,
		volatile unsigned* aSmSegments
	);

	// Compact
	template< typename T >
	static _CHAG_PP_DEV SizeType compact( 
		T* aOut, 
		const T& aVal, 
		bool aKeep,
		volatile SizeType* aSmPrefix 
	);

	template< typename T >
	static _CHAG_PP_DEV SizeType compact( 
		T* aOut, 
		const T& aV0, const T& aV1, 
		bool aKeep0, bool aKeep1,
		volatile SizeType* aSmPrefix
	);

	// Split
	template< typename T >
	static _CHAG_PP_DEV SizeType split( 
		T* aOut, 
		const T& aVal, 
		bool aKeep,
		volatile SizeType* aSm
	);
	template< typename T >
	static _CHAG_PP_DEV SizeType split( 
		T* aLeft, T* aRight, 
		const T& aVal, 
		bool aKeep,
		volatile SizeType* aSm
	);

	template< typename T >
	static _CHAG_PP_DEV SizeType split( 
		T* aOut, 
		const T& aV0, const T& aV1, 
		bool aKeep0, bool aKeep1, 
		volatile SizeType* aSm
	);
	template< typename T >
	static _CHAG_PP_DEV SizeType split( 
		T* aLeft, T* aRight, 
		const T& aV0, const T& aV1, 
		bool aKeep0, bool aKeep1,
		volatile SizeType* aSm
	);

	// 4-way split
	template< typename T, class Pred0, class Pred1 >
	static _CHAG_PP_DEV SizeType split4( 
		T* aOut, 
		const T& aVal, 
		const Pred0& aPred0, 
		const Pred1& aPred1, 
		volatile SizeType* aSm
	);
	template< typename T, class Pred0, class Pred1 >
	static _CHAG_PP_DEV SizeType split4( 
		T* aB0, T* aB1, T* aB2, T* aB3, 
		const T& aVal, 
		const Pred0& aPred0, 
		const Pred1& aPred1, 
		volatile SizeType* aSm
	);
	template< typename T, class Pred0, class Pred1 >
	static _CHAG_PP_DEV SizeType split4( 
		T* aBase, volatile SizeType* aOffsets, 
		const T& aVal, 
		const Pred0& aPred0, 
		const Pred1& aPred1, 
		volatile SizeType* aSm
	);

	template< typename T, class Pred0, class Pred1 >
	static _CHAG_PP_DEV SizeType split4( 
		T* aBase, volatile SizeType* aOffsets, 
		const T& aV0, const T& aV1, 
		const Pred0& aPred0, 
		const Pred1& aPred1, 
		volatile SizeType* aSm
	);

	// Utility
	static _CHAG_PP_DEV void barrier();

	protected:
		typedef typename KSetup::Env _Env;

	/* Shared memory requirements, in number of elements 
	 */
	public:
		enum { MEMORY_ELEMENTS_REDUCE = _Env::SIMD + _Env::SIMD/2 };
		enum { MEMORY_ELEMENTS_PREFIX = _Env::SIMD + _Env::SIMD/2 };

};

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#include "detail/unit.cu.inl"
#endif // _CHAG_PP_EXT_UNIT_CUH_
