/****************************************************************************/
/*!	\brief [chag::pp] Prefix Implementation
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
inline void prefix(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	T* aTotal
)
{
	Prefixer<
		aspect::prefix::InputType<T>
	>::prefix( aStart, aEnd, aOutput, aTotal, op::Add<T>() );
}
template< typename T, class Op > 
inline void prefix(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	T* aTotal,
	const Op& aOperator
)
{
	Prefixer<
		aspect::prefix::InputType<T>
	>::prefix( aStart, aEnd, aOutput, aTotal, aOperator );
}

template< typename T > 
inline void prefix_inclusive(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	T* aTotal
)
{
	Prefixer<
		aspect::prefix::InputType<T>,
		aspect::prefix::Inclusive< true >
	>::prefix( aStart, aEnd, aOutput, aTotal, op::Add<T>() );
}
template< typename T, class Op > 
inline void prefix_inclusive(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	T* aTotal,
	const Op& aOperator
)
{
	Prefixer<
		aspect::prefix::InputType<T>,
		aspect::prefix::Inclusive< true >
	>::prefix( aStart, aEnd, aOutput, aTotal, aOperator );
}


//--	Kernels				///{{{1///////////////////////////////////////////
template< class JSetup, typename T, class Op, class KParam >
_CHAG_PP_KRN void prefix_reduce( const T* aStart, T* aPartialReductions, 
	Op aOperator, KParam aParam )
{
//	typedef Range<T,JSetup> _Range; /* Causes: cudafe died due to foo */
#	define _Range Range<T,JSetup> /* HACK */

	CHAG_PP_KERNEL_SETUP( JSetup,
		none, none,
		jobs( aParam, aStart ),
		temp( 
			volatile T reduceBuffer[_Range::MEMORY_ELEMENTS_REDUCE];
		)
	);

	T partialReduction = _Range::reduce(
		aStart + _job.begin, aStart + _job.end, aOperator, _sm_temp.reduceBuffer
	);

	if( _Env::lane() == 0 )
		aPartialReductions[_job.id] = partialReduction;

	// Handle auxilary elements
	if( _Env::unit() == 0 )
	{
		const T* start = aStart + aParam.auxStart;
		const T* end = start + aParam.auxElements;

		T red = aOperator.identity();
		for( ; start < end; start += _Env::SIMD )
		{
			if( start + _Env::lane() < end )
			{
				T elem = start[_Env::lane()];
				red = aOperator( red, elem );
			}
		}

		T total = Unit<JSetup>::reduce( red, aOperator, _sm_temp.reduceBuffer );

		if( _Env::lane() == 0 )
			aPartialReductions[JSetup::JOB_COUNT] = total;
	}

#	undef _Range
}

template< unsigned Count, unsigned BuffSize, class KSetup, typename T, class Op > 
_CHAG_PP_KRN void prefix_offset( const T* aPartialReductions, T* aPartialOffsets, 
	T* aTotal, Op aOperator )
{
#	define _Range Range<T,KSetup>

	CHAG_PP_KERNEL_SETUP( KSetup,	
		none, none, none,
		temp(
			volatile T prefixBuffer[_Range::MEMORY_ELEMENTS_PREFIX];
		)
	);

	_Range::prefix(
		aPartialReductions, aPartialReductions + BuffSize, aPartialOffsets, 
		aOperator.identity(), aOperator, _sm_temp.prefixBuffer
	);

	if( _Env::lane() == 0 && aTotal )
		*aTotal = aPartialOffsets[Count+1];

#	undef _Range
}

template< class JSetup, typename T, class Op >
_CHAG_PP_KRN void prefix_prefix( const T* aStart, 
	const T* aPartialOffsets, T* aOut, Op aOperator, 
	typename JSetup::KernelParam aParam )
{
//	typedef Range<T,JSetup> _Range; /* Causes: cudafe died due to signal 6 */
#	define _Range Range<T,JSetup> /* HACK */

	CHAG_PP_KERNEL_SETUP( JSetup,
		jobs( aParam, aStart ),
		temp( 
			volatile T prefixBuffer[_Range::MEMORY_ELEMENTS_REDUCE];
		),
		persist(
			volatile T startOffset;
		),
		init(
			SizeType jobId = _job.id + _Env::lane();
			persist.startOffset = aPartialOffsets[jobId];
		)
	);

	_Range::prefix(
		aStart + _job.begin, aStart + _job.end,
		aOut + _job.begin, _sm_persist.startOffset, aOperator, 
		_sm_temp.prefixBuffer
	);
	
	// Handle auxilary elements
	if( _Env::unit() == 0 )
	{
		const T* start = aStart + aParam.auxStart;
		const T* end = start + aParam.auxElements;
		T* out = aOut + aParam.auxStart;

		T red = aPartialOffsets[JSetup::JOB_COUNT];
		for( ; start < end; start += _Env::SIMD )
		{
			T elem = aOperator.identity();

			if( start + _Env::lane() < end )
			{
				elem = start[_Env::lane()];
			}

			T res = aOperator( red,
				Unit<JSetup>::prefix( elem, aOperator, _sm_temp.prefixBuffer )
			);

			if( start + _Env::lane() < end )
			{
				out[_Env::lane()] = res;
			}

			red = aOperator( red, 
				Unit<JSetup>::prefix_total_get( _sm_temp.prefixBuffer )
			);

			out += _Env::SIMD;
		}
	}

#	undef _Range
}

template< class JSetup, typename T, class Op >
_CHAG_PP_KRN void prefix_prefix_inclusive( const T* aStart, 
	const T* aPartialOffsets, T* aOut, Op aOperator, 
	typename JSetup::KernelParam aParam )
{
//	typedef Range<T,JSetup> _Range; /* Causes: cudafe died due to signal 6 */
#	define _Range Range<T,JSetup> /* HACK */

	CHAG_PP_KERNEL_SETUP( JSetup,
		jobs( aParam, aStart ),
		temp( 
			volatile T prefixBuffer[_Range::MEMORY_ELEMENTS_REDUCE];
		),
		persist(
			volatile T startOffset;
		),
		init(
			SizeType jobId = _job.id + _Env::lane();
			persist.startOffset = aPartialOffsets[jobId];
		)
	);

	_Range::prefix_inclusive(
		aStart + _job.begin, aStart + _job.end,
		aOut + _job.begin, _sm_persist.startOffset, aOperator, 
		_sm_temp.prefixBuffer
	);
	
	// Handle auxilary elements
	if( _Env::unit() == 0 )
	{
		const T* start = aStart + aParam.auxStart;
		const T* end = start + aParam.auxElements;
		T* out = aOut + aParam.auxStart;

		T red = aPartialOffsets[JSetup::JOB_COUNT];
		for( ; start < end; start += _Env::SIMD )
		{
			T elem = aOperator.identity();

			if( start + _Env::lane() < end )
			{
				elem = start[_Env::lane()];
			}

			T res = aOperator( red,
				Unit<JSetup>::prefix_inclusive( elem, aOperator, _sm_temp.prefixBuffer )
			);

			if( start + _Env::lane() < end )
			{
				out[_Env::lane()] = res;
			}

			red = aOperator( red, 
				Unit<JSetup>::prefix_total_get( _sm_temp.prefixBuffer )
			);

			out += _Env::SIMD;
		}
	}

#	undef _Range
}

//--	Prefixer			///{{{1///////////////////////////////////////////
NTA_TEMPLATE_PREAMBLE template< class Op >
void Prefixer<NTA_TEMPLATE_ARGUMENTS>::prefix(
	const input_type* aStart, const input_type* aEnd,
	output_type* aOutput, output_type* aTotal, const Op& aOperator,
	output_type* aPartialReductions, output_type* aPartialOffsets
)
{
	JSetup setup; setup.setup( SizeType(aEnd-aStart) );

	output_type* partialRed = aPartialReductions;
	if( !partialRed )
	{
		/*FIXME*/
		partialRed = (output_type*)detail::OffsetBuffer::count_buffer( 
			JSetup::JOB_BUFFER_SIZE*sizeof(output_type)/sizeof(SizeType)
		);
	}

	output_type* partialOffs = aPartialOffsets;
	if( !partialOffs )
	{
		/*FIXME*/
		partialOffs = (output_type*)detail::OffsetBuffer::offset_buffer( 
			JSetup::JOB_BUFFER_SIZE*sizeof(output_type)/sizeof(SizeType)
		);
	}

	dim3 cub = KSetup::layout_cuda_blocks();
	dim3 cut = KSetup::layout_cuda_threads();

	prefix_reduce<JSetup><<<cub,cut>>>( 
		aStart, partialRed, aOperator, setup.kernelParam() 
	);
	prefix_offset<
		JSetup::JOB_COUNT, JSetup::JOB_BUFFER_SIZE,
		KernelSetupBlock<1,512>
	><<<1,512>>>( partialRed, partialOffs, aTotal, aOperator );

	if( INCLUSIVE )
	{
		prefix_prefix_inclusive<JSetup><<<cub,cut>>>( 
			aStart, partialOffs, aOutput, aOperator, setup.kernelParam() 
		);
	}
	else
	{
		prefix_prefix<JSetup><<<cub,cut>>>( 
			aStart, partialOffs, aOutput, aOperator, setup.kernelParam() 
		);
	}
}

//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
CHAG_PP_LEAVE_NAMESPACE()
