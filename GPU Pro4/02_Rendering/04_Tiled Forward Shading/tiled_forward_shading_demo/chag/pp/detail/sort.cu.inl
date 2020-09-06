/****************************************************************************/
/*!	\brief [chag::pp] Sort Implementation
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
inline T* sort(
	const T* aStart, 
	const T* aEnd,
	T* aPing, T* aPong
)
{
	return Sorter<
		aspect::sort::InputType<T>
	>::sort( aStart, aEnd, aPing, aPong );
}
template< 
	unsigned Low, 
	unsigned High, 
	template<unsigned,typename> class Pred, 
	typename T 
> 
inline T* sort(
	const T* aStart, 
	const T* aEnd,
	T* aPing, T* aPong
)
{
	return Sorter<
		aspect::sort::InputType<T>,
		aspect::sort::Predicate<Pred>,
		aspect::sort::LimitLow<Low>,
		aspect::sort::LimitHigh<High>
	>::sort( aStart, aEnd, aPing, aPong );
}

//--	Kernels - 2-way		///{{{1///////////////////////////////////////////
template< class JSetup, typename T, class Pred, class KParam >
_CHAG_PP_KRN void sort_count( const T* aStart, SizeType* aCounts, 
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
_CHAG_PP_KRN void sort_prefix( const SizeType* aCounts, SizeType* aOffsets )
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
}

template< class JSetup, bool Stage, typename T, class Pred >
_CHAG_PP_KRN void sort_move( const T* aStart, 
	const SizeType* aCounts, const SizeType* aOffsets,
	T* aOut, Pred aPredicate, typename JSetup::KernelParam aParam )
{
//	typedef Range<T,JSetup> _Range; /* Causes: cudafe died due to signal 6 */
#	define _Range Range<T,JSetup> /* HACK */

	enum { CanStage = Stage ? 1 : 0 };

	CHAG_PP_KERNEL_SETUP( JSetup,
		jobs( aParam, aStart ),
		temp( 
			volatile SizeType prefixBuffer[_Range::MEMORY_ELEMENTS_REDUCE];
			volatile T stageBuffer[CanStage * _Range::MEMORY_ELEMENTS_STAGE];
		),
		persist(
			volatile SizeType total;
			volatile SizeType startOffset;
		),
		init(
			SizeType jobId = _job.id + _Env::lane();
			persist.startOffset = aOffsets[jobId];

			persist.total = aOffsets[JSetup::JOB_COUNT+1];
		)
	);


	if( !Stage )
	{
		_Range::split_scatter(
			aStart + _job.begin, aStart + _job.end,
			aOut + _sm_persist.startOffset,
			aOut + _sm_persist.total + _job.begin - _sm_persist.startOffset,
			aPredicate, _sm_temp.prefixBuffer
		);
	}
	else
	{
		_Range::split_stage(
			aStart + _job.begin, aStart + _job.end,
			aOut + _sm_persist.startOffset,
			aOut + _sm_persist.total + _job.begin - _sm_persist.startOffset,
			aPredicate, _sm_temp.stageBuffer, _sm_temp.prefixBuffer
		);
	}
	
	// Handle auxilary elements
	if( _Env::unit() == 0 )
	{
		const T* start = aStart + aParam.auxStart;
		const T* end = start + aParam.auxElements;

		SizeType offs = aOffsets[JSetup::JOB_COUNT];

		T* left = aOut + offs;
		T* right = aOut + _sm_persist.total + aParam.auxStart - offs;

		for( ; start < end; start += _Env::SIMD )
		{
			T elem;
			bool valid = false;

			if( start + _Env::lane() < end )
			{
				elem = start[_Env::lane()];
				valid = aPredicate(elem);
			}

			SizeType o = Unit<JSetup>::prefix( valid ? 1u:0u, op::Add<SizeType>(),
				_sm_temp.prefixBuffer
			);
 			SizeType t = Unit<JSetup>::prefix_total_get( _sm_temp.prefixBuffer );

			if( start + _Env::lane() < end )
			{
				if( valid )
					left[o] = elem;
				else
					right[_Env::lane() - o] = elem;
			}

			SizeType workaround = _Env::SIMD -t;
 
			left += t;
			right += workaround;
		}
	}

#undef _Range
}

//--	Kernels - 4-way		///{{{1///////////////////////////////////////////
template< class JSetup, typename T, class Pred0, class Pred1, class KParam >
_CHAG_PP_KRN void sort_count_4( const T* aStart, SizeType* aCounts, 
	Pred0 aPred0, Pred1 aPred1, KParam aParam )
{
//	typedef Range<T,JSetup> _Range; /* Causes: cudafe died due to foo */
#	define _Range Range<T,JSetup> /* HACK */

	CHAG_PP_KERNEL_SETUP( JSetup,
		none, none,
		jobs( aParam, aStart ),
		temp( 
			volatile SizeType countBuffer[4*_Env::SIMD];
			volatile SizeType reduceBuffer[_Range::MEMORY_ELEMENTS_REDUCE];
		)
	);

	typename _Range::SizeType4 counts = _Range::count4(
		aStart + _job.begin, aStart + _job.end, aPred0, aPred1, 
		_sm_temp.countBuffer, _sm_temp.reduceBuffer
	);

	if( _Env::lane() == 0 )
	{
		aCounts[_job.id+0*JSetup::JOB_BUFFER_SIZE] = counts.x;
		aCounts[_job.id+1*JSetup::JOB_BUFFER_SIZE] = counts.y;
		aCounts[_job.id+2*JSetup::JOB_BUFFER_SIZE] = counts.z;
		aCounts[_job.id+3*JSetup::JOB_BUFFER_SIZE] = counts.w;
	}

	// Handle auxilary elements
	if( _Env::unit() == 0 )
	{
		const T* start = aStart + aParam.auxStart;
		const T* end = start + aParam.auxElements;

		_sm_temp.countBuffer[0*_Env::SIMD + _Env::lane()] = 0;
		_sm_temp.countBuffer[1*_Env::SIMD + _Env::lane()] = 0;
		_sm_temp.countBuffer[2*_Env::SIMD + _Env::lane()] = 0;
		_sm_temp.countBuffer[3*_Env::SIMD + _Env::lane()] = 0;

		for( ; start < end; start += _Env::SIMD )
		{
			if( start + _Env::lane() < end )
			{
				T elem = start[_Env::lane()];

				bool pred0 = aPred0(elem);
				bool pred1 = aPred1(elem);

				SizeType idx = (pred0 ? 0u : _Env::SIMD) + (pred1 ? 0u : 2*_Env::SIMD);
				++_sm_temp.countBuffer[idx + _Env::lane()];
			}
		}

		SizeType pc0, pc1, pc2, pc3;
		pc0 = _sm_temp.reduceBuffer[0*_Env::SIMD + _Env::lane()];
		pc1 = _sm_temp.reduceBuffer[1*_Env::SIMD + _Env::lane()];
		pc2 = _sm_temp.reduceBuffer[2*_Env::SIMD + _Env::lane()];
		pc3 = _sm_temp.reduceBuffer[3*_Env::SIMD + _Env::lane()];
		Unit<JSetup>::barrier();

		typename _Range::SizeType4 sum;
		sum.x = Unit<JSetup>::reduce( pc0, op::Add<SizeType>(), _sm_temp.reduceBuffer );
		sum.y = Unit<JSetup>::reduce( pc1, op::Add<SizeType>(), _sm_temp.reduceBuffer );
		sum.z = Unit<JSetup>::reduce( pc2, op::Add<SizeType>(), _sm_temp.reduceBuffer );
		sum.w = Unit<JSetup>::reduce( pc3, op::Add<SizeType>(), _sm_temp.reduceBuffer );

		if( _Env::lane() == 0 )
		{
			aCounts[JSetup::JOB_COUNT+0*JSetup::JOB_BUFFER_SIZE] = sum.x;
			aCounts[JSetup::JOB_COUNT+1*JSetup::JOB_BUFFER_SIZE] = sum.y;
			aCounts[JSetup::JOB_COUNT+2*JSetup::JOB_BUFFER_SIZE] = sum.z;
			aCounts[JSetup::JOB_COUNT+3*JSetup::JOB_BUFFER_SIZE] = sum.w;
		}
	}
#undef _Range
}

template< unsigned Count, unsigned BuffSize, class KSetup > 
_CHAG_PP_KRN void sort_prefix_4( const SizeType* aCounts, SizeType* aOffsets )
{
	typedef Unit<KSetup> _Unit;
	typedef Range<SizeType,KSetup> _Range;

	CHAG_PP_KERNEL_SETUP( KSetup,	
		none, none, none,
		temp(
			volatile SizeType prefixBuffer[_Range::MEMORY_ELEMENTS_PREFIX];
		)
	);
	
	SizeType total = 0;

	// way-1
	_Range::prefix(
		aCounts + 0*BuffSize, aCounts + 1*BuffSize, aOffsets + 0*BuffSize, total, 
		op::Add<SizeType>(), _sm_temp.prefixBuffer
	);

	total = aOffsets[0*BuffSize+Count+1];
	_Unit::barrier();

	// way-2
	_Range::prefix(
		aCounts + 1*BuffSize, aCounts + 2*BuffSize, aOffsets + 1*BuffSize, total, 
		op::Add<SizeType>(), _sm_temp.prefixBuffer
	);

	total = aOffsets[1*BuffSize+Count+1];
	_Unit::barrier();

	// way-3
	_Range::prefix(
		aCounts + 2*BuffSize, aCounts + 3*BuffSize, aOffsets + 2*BuffSize, total, 
		op::Add<SizeType>(), _sm_temp.prefixBuffer
	);

	total = aOffsets[2*BuffSize+Count+1];
	_Unit::barrier();

	// way-4
	_Range::prefix(
		aCounts + 3*BuffSize, aCounts + 4*BuffSize, aOffsets + 3*BuffSize, total, 
		op::Add<SizeType>(), _sm_temp.prefixBuffer
	);
}

template< class JSetup, bool Stage, typename T, class Pred0, class Pred1 >
_CHAG_PP_KRN void sort_move_4( const T* aStart, 
	const SizeType* aCounts, const SizeType* aOffsets,
	T* aOut, Pred0 aPred0, Pred1 aPred1, typename JSetup::KernelParam aParam )
{
//	typedef Range<T,JSetup> _Range; /* Causes: cudafe died due to signal 6 */
#	define _Range Range<T,JSetup> /* HACK */

	enum { CanStage = Stage ? 1 : 0 };

	CHAG_PP_KERNEL_SETUP( JSetup,
		jobs( aParam, aStart ),
		temp( 
			volatile SizeType prefixBuffer[_Range::MEMORY_ELEMENTS_REDUCE];
		),
		persist(
			volatile SizeType startOffsets[4];
		),
		init(
			SizeType jobId = _job.id + _Env::lane();
			persist.startOffsets[0] = aOffsets[jobId + 0*JSetup::JOB_BUFFER_SIZE];
			persist.startOffsets[1] = aOffsets[jobId + 1*JSetup::JOB_BUFFER_SIZE];
			persist.startOffsets[2] = aOffsets[jobId + 2*JSetup::JOB_BUFFER_SIZE];
			persist.startOffsets[3] = aOffsets[jobId + 3*JSetup::JOB_BUFFER_SIZE];
		)
	);

	_Range::split4_scatter(
		aStart + _job.begin, aStart + _job.end,
		aOut, _sm_persist.startOffsets,
		aPred0, aPred1, _sm_temp.prefixBuffer
	);

	// Handle auxilary elements
	if( _Env::unit() == 0 )
	{
		const T* start = aStart + aParam.auxStart;
		const T* end = start + aParam.auxElements;

		if( _Env::lane() < 4 )
		{
			_sm_persist.startOffsets[_Env::lane()] =
				aOffsets[JSetup::JOB_COUNT + _Env::lane()*JSetup::JOB_BUFFER_SIZE];
		}
		Unit<JSetup>::barrier();

		for( ; start < end; start += _Env::SIMD )
		{
			T elem;
			SizeType idx = 0, inc = 0;

			if( start + _Env::lane() < end )
			{
				elem = start[_Env::lane()];

				bool pred0 = aPred0(elem);
				bool pred1 = aPred1(elem);

				idx = (pred0 ? 0u : 1u) + (pred1 ? 0u : 2u);
				inc = 1u << (idx << 3u); // WARN: max/bucket < 255
			}

			SizeType bdest = Unit<JSetup>::prefix( inc, op::Add<SizeType>(),
				_sm_temp.prefixBuffer
			);
 			SizeType count = Unit<JSetup>::prefix_total_get( _sm_temp.prefixBuffer );
			SizeType dest = _sm_persist.startOffsets[idx] + 
				(0xFFu & (bdest >> (idx << 3u)));

			if( start + _Env::lane() < end )
			{
				aOut[dest] = elem;
			}
			Unit<JSetup>::barrier();
 
			if( _Env::lane() < 4 )
			{
				_sm_persist.startOffsets[_Env::lane()] +=
					(0xFFu & (count >> (_Env::lane() << 3u)));
			}
			Unit<JSetup>::barrier();
		}
	}
#undef _Range
}

//--	Helpers				///{{{1///////////////////////////////////////////
namespace detail
{
	template< 
		bool _4, bool Stage,
		unsigned Low, 
		unsigned High, 
		template <unsigned> class Pred,
		class JSetup
	> struct SortPassIterator
	{
		template< typename T, class KParam >
		static T* run( T* aPing, T* aPong, SizeType* aCBuff, SizeType* aOBuff, 
			const KParam& aParam, const dim3& aB, const dim3& aT )
		{
			sort_count<JSetup><<<aB,aT>>>( 
				aPing, aCBuff, Pred<Low>(), aParam 
			);
			sort_prefix<
				JSetup::JOB_COUNT, JSetup::JOB_BUFFER_SIZE,
				KernelSetupBlock<1,512>
			><<<1,512>>>( aCBuff, aOBuff );
			sort_move<JSetup,Stage><<<aB,aT>>>( 
				aPing, aCBuff, aOBuff, aPong, Pred<Low>(), aParam 
			);

			return SortPassIterator<
				false, Stage,
				Low+1, High,
				Pred, JSetup
			>::run( aPong, aPing, aCBuff, aOBuff, aParam, aB, aT );
		}
	};

	template<
		bool Stage, 
		unsigned Low,
		unsigned High,
		template <unsigned> class Pred,
		class JSetup
	> struct SortPassIterator<true,Stage,Low,High,Pred,JSetup>
	{
		template< typename T, class KParam >
		static T* run( T* aPing, T* aPong, SizeType* aCBuff, SizeType* aOBuff, 
			const KParam& aParam, const dim3& aB, const dim3& aT )
		{
			sort_count_4<JSetup><<<aB,aT>>>( 
				aPing, aCBuff, Pred<Low>(), Pred<Low+1>(), aParam 
			);
			sort_prefix_4<
				JSetup::JOB_COUNT, JSetup::JOB_BUFFER_SIZE,
				KernelSetupBlock<1,512>
			><<<1,512>>>( aCBuff, aOBuff );
			sort_move_4<JSetup,Stage><<<aB,aT>>>( 
				aPing, aCBuff, aOBuff, aPong, Pred<Low>(), Pred<Low+1>(), aParam 
			);

			return SortPassIterator< 
				(High-Low >= 4), Stage,
				Low+2, High,
				Pred, JSetup
			>::run( aPong, aPing, aCBuff, aOBuff, aParam, aB, aT );
		}

	};

	template< 
		bool _4, bool Stage, 
		unsigned High, 
		template <unsigned> class Pred,
		class JSetup
	> struct SortPassIterator<_4,Stage,High,High,Pred,JSetup>
	{
		template< typename T, class KParam >
		static T* run( T* aPing, T*, SizeType*, SizeType*, 
			const KParam&, const dim3&, const dim3& )
		{
			return aPing;
		}
	};
}

//--	Compactor			///{{{1///////////////////////////////////////////
NTA_TEMPLATE_PREAMBLE typename Sorter<NTA_TEMPLATE_ARGUMENTS>::output_type* 
Sorter<NTA_TEMPLATE_ARGUMENTS>::sort(
	const input_type* aStart, const input_type* aEnd,
	output_type* aPing, output_type* aPong, 
	SizeType* aCountBuffer, SizeType* aOffsetBuffer
)
{
	if( unsigned(LIMIT_LOW) >= unsigned(LIMIT_HIGH) )
	{
		return 0;
	}

	JSetup setup; setup.setup( SizeType(aEnd-aStart) );

	SizeType* countBuffer = aCountBuffer;
	if( !countBuffer )
		countBuffer = detail::OffsetBuffer::count_buffer( JSetup::JOB_BUFFER_SIZE*4 );

	SizeType* offsetBuffer = aOffsetBuffer;
	if( !offsetBuffer )
		offsetBuffer = detail::OffsetBuffer::offset_buffer( JSetup::JOB_BUFFER_SIZE*4 );

	dim3 cub = KSetup::layout_cuda_blocks();
	dim3 cut = KSetup::layout_cuda_threads();

	// first pass: move from input buffer to ping buffer
	if( LIMIT_HIGH - LIMIT_LOW < 2 )
	{
		sort_count<JSetup><<<cub,cut>>>(
			aStart, countBuffer, Pred<LIMIT_LOW>(), setup.kernelParam()
		);
		sort_prefix<
			JSetup::JOB_COUNT, JSetup::JOB_BUFFER_SIZE,
			KernelSetupBlock<1,512>
		><<<1,512>>>( countBuffer, offsetBuffer );
		sort_move<JSetup,STAGE><<<cub,cut>>>(
			aStart, countBuffer, offsetBuffer, aPing, Pred<LIMIT_LOW>(), 
			setup.kernelParam()
		);

		return detail::SortPassIterator<
			false, STAGE,
			LIMIT_LOW+1, LIMIT_HIGH, 
			Pred,
			JSetup
		>::run(
			aPing, aPong, countBuffer, offsetBuffer, setup.kernelParam(), cub, cut
		);
	}
	else
	{
		sort_count_4<JSetup><<<cub,cut>>>( 
			aStart, countBuffer, Pred<LIMIT_LOW>(), Pred<LIMIT_LOW+1>(), 
			setup.kernelParam() 
		);
		sort_prefix_4<
			JSetup::JOB_COUNT, JSetup::JOB_BUFFER_SIZE,
			KernelSetupBlock<1,512>
		><<<1,512>>>( countBuffer, offsetBuffer );
		sort_move_4<JSetup,STAGE><<<cub,cut>>>( 
			aStart, countBuffer, offsetBuffer, aPing, Pred<LIMIT_LOW>(), 
			Pred<LIMIT_LOW+1>(), setup.kernelParam() 
		);

		return detail::SortPassIterator<
			(LIMIT_HIGH-LIMIT_LOW) >= 4, STAGE,
			LIMIT_LOW+2, LIMIT_HIGH, 
			Pred,
			JSetup
		>::run(
			aPing, aPong, countBuffer, offsetBuffer, setup.kernelParam(), cub, cut
		);

	}
}

//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
CHAG_PP_LEAVE_NAMESPACE()
