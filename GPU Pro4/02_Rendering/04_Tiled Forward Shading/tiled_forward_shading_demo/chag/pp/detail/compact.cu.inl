/****************************************************************************/
/*!	\brief [chag::pp] Compaction Implementation
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

//--//////////////////////////////////////////////////////////////////////////
CHAG_PP_ENTER_NAMESPACE()
//--	compact()			///{{{1///////////////////////////////////////////
template< typename T > 
inline void compact(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	SizeType* aNumValid
)
{
	Compactor<
		aspect::compact::InputType<T>
	>::compact( aStart, aEnd, aOutput, aNumValid, pred::Nonzero<T>() );
}
template< typename T, class Predicate > 
inline void compact(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	SizeType* aNumValid,
	const Predicate& aPredicate
)
{
	Compactor<
		aspect::compact::InputType<T>
	>::compact( aStart, aEnd, aOutput, aNumValid, aPredicate );
}

//--	Kernels				///{{{1///////////////////////////////////////////
template< class JSetup, typename T, class Pred, class KParam >
_CHAG_PP_KRN void compact_count( const T* aStart, SizeType* aCounts, 
	Pred aPredicate, KParam aParam )
{
//	typedef Range<T,JSetup> _Range; /* Causes: cudafe died due to foo */
#	define _Range Range<T,JSetup> /* HACK */

	CHAG_PP_KERNEL_SETUP( JSetup,
		none, none,
		jobs( aParam, aStart ),
		temp( 
			volatile SizeType reduceBuffer[_Range::MEMORY_ELEMENTS_REDUCE];
		)
	);

	SizeType count = _Range::count(
		aStart + _job.begin, aStart + _job.end, aPredicate, _sm_temp.reduceBuffer
	);

	if( _Env::lane() == 0 )
		aCounts[_job.id] = count;

	// Handle auxilary elements
	if( _Env::unit() == 0 )
	{
		const T* start = aStart + aParam.auxStart;
		const T* end = start + aParam.auxElements;

		SizeType count = 0;
		for( ; start < end; start += _Env::SIMD )
		{
			if( start + _Env::lane() < end )
			{
				T elem = start[_Env::lane()];
				count += aPredicate(elem) ? 1 : 0;
			}
		}

		SizeType total = Unit<JSetup>::reduce( count, op::Add<SizeType>(),
			_sm_temp.reduceBuffer
		);

		if( _Env::lane() == 0 )
			aCounts[JSetup::JOB_COUNT] = total;
	}
#undef _Range
}

template< unsigned Count, unsigned BuffSize, class KSetup > 
_CHAG_PP_KRN void compact_prefix( const SizeType* aCounts, SizeType* aOffsets, 
	SizeType* aNumValid )
{
	typedef Range<SizeType,KSetup> _Range;

	CHAG_PP_KERNEL_SETUP( KSetup,	
		none, none, none,
		temp(
			volatile SizeType prefixBuffer[_Range::MEMORY_ELEMENTS_PREFIX];
		)
	);

	_Range::prefix(
		aCounts, aCounts + BuffSize, aOffsets, 0, op::Add<SizeType>(), 
		_sm_temp.prefixBuffer
	);

	if( _Env::lane() == 0 && aNumValid )
		*aNumValid = aOffsets[Count+1];
}

template< class JSetup, unsigned SRatio, typename T, class Pred >
_CHAG_PP_KRN void compact_move( const T* aStart, 
	const SizeType* aCounts, const SizeType* aOffsets,
	T* aOut, Pred aPredicate, typename JSetup::KernelParam aParam )
{
//	typedef Range<T,JSetup> _Range; /* Causes: cudafe died due to signal 6 */
#	define _Range Range<T,JSetup> /* HACK */

	enum { CanStage = SRatio > 100 ? 0 : 1 };
	enum { StageBufferSize = CanStage ? _Range::MEMORY_ELEMENTS_STAGE : 1 };

	CHAG_PP_KERNEL_SETUP( JSetup,
		jobs( aParam, aStart ),
		temp( 
			volatile SizeType prefixBuffer[_Range::MEMORY_ELEMENTS_REDUCE];
			volatile T stageBuffer[StageBufferSize];
		),
		persist(
			volatile bool shouldStage;
			volatile SizeType startOffset;
		),
		init(
			SizeType jobId = _job.id + _Env::lane();

			persist.startOffset = aOffsets[jobId];
			persist.shouldStage = false;

			if( CanStage )
			{
				SizeType rcount = aCounts[jobId];
				if( rcount <= (_job.end-_job.begin) * (SRatio/100.0f) )
					persist.shouldStage = true;
			}
		)
	);

	if( !_sm_persist.shouldStage )
	{
		_Range::compact_scatter(
			aStart + _job.begin, aStart + _job.end,
			aOut + _sm_persist.startOffset,
			aPredicate, _sm_temp.prefixBuffer
		);
	}
	else
	{
		_Range::compact_stage(
			aStart + _job.begin, aStart + _job.end,
			aOut + _sm_persist.startOffset,
			aPredicate, _sm_temp.stageBuffer, _sm_temp.prefixBuffer
		);
	}
	
	// Handle auxilary elements
	if( _Env::unit() == 0 )
	{
		const T* start = aStart + aParam.auxStart;
		const T* end = start + aParam.auxElements;
		T* out = aOut + aOffsets[JSetup::JOB_COUNT];

		for( ; start < end; start += _Env::SIMD )
		{
			T elem;
			bool valid = false;

			if( start + _Env::lane() < end )
			{
				elem = start[_Env::lane()];
				valid = aPredicate(elem);
			}

			T* dest = out + Unit<JSetup>::prefix( valid ? 1u:0u, op::Add<SizeType>(),
				_sm_temp.prefixBuffer
			);

			if( valid )
				*dest = elem;

			out += Unit<JSetup>::prefix_total_get( _sm_temp.prefixBuffer );
		}
	}

#undef _Range
}

//--	Compactor			///{{{1///////////////////////////////////////////
NTA_TEMPLATE_PREAMBLE template< class Predicate >
void Compactor<NTA_TEMPLATE_ARGUMENTS>::compact(
	const input_type* aStart, const input_type* aEnd,
	output_type* aOutput, SizeType* aNumValid, const Predicate& aPredicate,
	SizeType* aCountBuffer, SizeType* aOffsetBuffer
)
{
	JSetup setup; setup.setup( SizeType(aEnd-aStart) );

	SizeType* countBuffer = aCountBuffer;
	if( !countBuffer )
		countBuffer = detail::OffsetBuffer::count_buffer( JSetup::JOB_BUFFER_SIZE );

	SizeType* offsetBuffer = aOffsetBuffer;
	if( !offsetBuffer )
		offsetBuffer = detail::OffsetBuffer::offset_buffer( JSetup::JOB_BUFFER_SIZE );

	dim3 cub = KSetup::layout_cuda_blocks();
	dim3 cut = KSetup::layout_cuda_threads();

	compact_count<JSetup><<<cub,cut>>>( 
		aStart, countBuffer, aPredicate, setup.kernelParam() 
	);
	compact_prefix<
		JSetup::JOB_COUNT, JSetup::JOB_BUFFER_SIZE,
		KernelSetupBlock<1,512>
	><<<1,512>>>( countBuffer, offsetBuffer, aNumValid );
	compact_move<JSetup,SRATIO><<<cub,cut>>>( 
		aStart, countBuffer, offsetBuffer, aOutput, aPredicate, setup.kernelParam() 
	);
}

//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
CHAG_PP_LEAVE_NAMESPACE()
