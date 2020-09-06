/****************************************************************************/
/*!	\brief [chag::pp] Reduction Implementation
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
inline void reduce(
	const T* aStart, 
	const T* aEnd,
	T* aOutput
)
{
	Reducer<
		aspect::reduce::InputType<T>
	>::reduce( aStart, aEnd, aOutput, op::Add<T>() );
}
template< typename T, class Op > 
inline void reduce(
	const T* aStart, 
	const T* aEnd,
	T* aOutput,
	const Op& aOperator
)
{
	Reducer<
		aspect::reduce::InputType<T>
	>::reduce( aStart, aEnd, aOutput, aOperator );
}

//--	Kernels				///{{{1///////////////////////////////////////////
template< class JSetup, typename T, class Op, class KParam >
_CHAG_PP_KRN void reduce_partial( const T* aStart, T* aPartialReductions, 
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

	T result = _Range::reduce(
		aStart + _job.begin, aStart + _job.end, aOperator, _sm_temp.reduceBuffer
	);

	if( _Env::lane() == 0 )
		aPartialReductions[_job.id] = result;

	// Handle auxilary elements
	if( _Env::unit() == 0 )
	{
		const T* start = aStart + aParam.auxStart;
		const T* end = start + aParam.auxElements;

		T result = aOperator.identity();
		for( ; start < end; start += _Env::SIMD )
		{
			if( start + _Env::lane() < end )
			{
				T elem = start[_Env::lane()];
				result = aOperator(result, elem);
			}
		}

		T total = Unit<JSetup>::reduce( result, aOperator, _sm_temp.reduceBuffer );

		if( _Env::lane() == 0 )
			aPartialReductions[JSetup::JOB_COUNT] = total;
	}

#undef _Range
}

template< unsigned Count, unsigned BuffSize, class KSetup, typename T, class Op > 
_CHAG_PP_KRN void reduce_final( const T* aPartials, T* aOutput, Op aOperator )
{
//	typedef Range<T,KSetup> _Range;
#	define _Range Range<T,KSetup>

	CHAG_PP_KERNEL_SETUP( KSetup,	
		none, none, none,
		temp(
			volatile T reduceBuffer[_Range::MEMORY_ELEMENTS_REDUCE];
		)
	);

	const T* start = aPartials;
	const T* end = aPartials + Count +1;

	T result = aOperator.identity();
	for( ; start < end; start += _Env::SIMD )
	{
		if( start + _Env::lane() < end )
		{
			T elem = start[_Env::lane()];
			result = aOperator(result, elem);
		}
	}

	T total = Unit<KSetup>::reduce( result, aOperator, _sm_temp.reduceBuffer );

	if( _Env::lane() == 0 )
		*aOutput = total;
#undef _Range
}


//--	Compactor			///{{{1///////////////////////////////////////////
NTA_TEMPLATE_PREAMBLE template< class Op >
void Reducer<NTA_TEMPLATE_ARGUMENTS>::reduce(
	const input_type* aStart, const input_type* aEnd,
	output_type* aOutput, const Op& aOperator,
	input_type* aPartialBuffer
)
{
	JSetup setup; setup.setup( SizeType(aEnd-aStart) );

	input_type* partialBuff = aPartialBuffer;
	if( !partialBuff )
	{
		//FIXME!
		partialBuff = (input_type*)detail::OffsetBuffer::count_buffer( 
			JSetup::JOB_BUFFER_SIZE * sizeof(input_type)/sizeof(SizeType) /*FIXME*/
		);
	}

	dim3 cub = KSetup::layout_cuda_blocks();
	dim3 cut = KSetup::layout_cuda_threads();

	reduce_partial<JSetup><<<cub,cut>>>( 
		aStart, partialBuff, aOperator, setup.kernelParam() 
	);
	reduce_final<
		JSetup::JOB_COUNT, JSetup::JOB_BUFFER_SIZE,
		KernelSetupBlock<1,512>
	><<<1,512>>>( partialBuff, aOutput, aOperator );
}


//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
CHAG_PP_LEAVE_NAMESPACE()
