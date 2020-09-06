/****************************************************************************/
/*!	\brief chag::pp Unit<> implementations
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
//--	Unit : utility		///{{{1///////////////////////////////////////////
template< class KSetup > 
_CHAG_PP_DEV void Unit<KSetup>::barrier()
{
	detail::conditional_barrier<_Env::SIMD,_Env::SYNC>();
}

//--	Unit : reduce		///{{{1///////////////////////////////////////////
namespace detail
{
	template< typename T, class Op, unsigned N, unsigned SIMD, unsigned SYNC > 
	struct BlockReduceHelper
	{
		static _CHAG_PP_DEV void eval( unsigned aTID, const Op& aOp, volatile T* aSm )
		{
			T temp = aOp( smX_get_striped<SIMD+SIMD/2>(aSm,aTID), smX_get_striped<SIMD+SIMD/2>(aSm,aTID+N) );
			detail::conditional_barrier<N*2,SYNC>();
			
			smX_set_striped<SIMD+SIMD/2>( aSm, aTID, temp );
			detail::conditional_barrier<N*2,SYNC>();

			BlockReduceHelper<T,Op,N/2u,SIMD,SYNC>::eval( aTID, aOp, aSm ); 
		}
	};
	template< typename T, class Op, unsigned SIMD, unsigned SYNC > 
	struct BlockReduceHelper<T,Op,1u,SIMD,SYNC>
	{
		static _CHAG_PP_DEV void eval( unsigned aTID, const Op& aOp, volatile T* aSm )
		{
			smX_set_striped<SIMD+SIMD/2>( aSm, aTID, aOp( smX_get_striped<SIMD+SIMD/2>(aSm,aTID), smX_get_striped<SIMD+SIMD/2>(aSm, aTID+1u) ) );
		}
	};
}

template< class KSetup > template< typename T, class Op >
_CHAG_PP_DEV T Unit<KSetup>::reduce( 
	const T& aVal, const Op& aOp, volatile T* aSmReduce
)
{
	//detail::sm_set( aSmReduce, _Env::lane(), aVal );
	detail::smX_set_striped<_Env::SIMD+_Env::SIMD/2>( aSmReduce, _Env::lane(), aVal );
	barrier();

	detail::BlockReduceHelper< T, Op, _Env::SIMD/2u, _Env::SIMD, _Env::SYNC >::eval( 
		_Env::lane(), aOp, aSmReduce
	);
	barrier();

	T ret = detail::smX_get_striped<_Env::SIMD+_Env::SIMD/2>( aSmReduce, 0u );
	barrier();

	return ret;
}

//--	Unit : prefix		///{{{1///////////////////////////////////////////
namespace detail
{
	template< typename T, class Op, unsigned N, unsigned M, bool E, unsigned SYNC >
	struct BlockPrefixHelper
	{
		static _CHAG_PP_DEV T eval( const T& aV, const Op& aOp, 
			unsigned aTID, volatile T* aSm )
		{
			T temp = aOp( aV, sm_get( aSm, aTID-M ) ); 
			detail::conditional_barrier<N,SYNC>();

			sm_set( aSm, aTID, temp );
			detail::conditional_barrier<N,SYNC>();

			return BlockPrefixHelper<T,Op,N,M*2,N==M*4,SYNC>::eval( 
				temp, aOp, aTID, aSm 
			);
		}
	};
	template< typename T, class Op, unsigned N, unsigned M, unsigned SYNC >
	struct BlockPrefixHelper<T,Op,N,M,true,SYNC>
	{
		static _CHAG_PP_DEV T eval( const T& aV, const Op& aOp, 
			unsigned aTID, volatile T* aSm )
		{
			T val = aOp( aV, sm_get( aSm, aTID-M ) );
			sm_set( aSm, aTID, val );
			return val;
		}
	};
}

template< class KSetup > template< typename T, class Op >
_CHAG_PP_DEV T Unit<KSetup>::prefix( 
	const T& aVal, const Op& aOp, volatile T* aSmPrefix )
{
	volatile T* tmp = aSmPrefix + (_Env::SIMD/2);

	detail::sm_set( aSmPrefix, _Env::lane(), aOp.identity() );
	barrier();

	detail::sm_set( tmp, _Env::lane(), aVal );
	barrier();

#if 0
	T res = detail::BlockPrefixHelper<T,Op,_Env::SIMD,1,false,_Env::SYNC>::eval(
		aVal, aOp, _Env::lane(), tmp
	) - aVal;
	barrier();

	return res;
#else
	detail::BlockPrefixHelper<T,Op,_Env::SIMD,1,false,_Env::SYNC>::eval(
		aVal, aOp, _Env::lane(), tmp
	);
	barrier();

	//T res = tmp[_Env::lane()-1];
	T res = detail::sm_get( tmp, _Env::lane()-1 );
	barrier();

	return res;
#endif
}
template< class KSetup > template< typename T, class Op >
_CHAG_PP_DEV T Unit<KSetup>::prefix_inclusive( 
	const T& aVal, const Op& aOp, volatile T* aSmPrefix )
{
	volatile T* tmp = aSmPrefix + (_Env::SIMD/2);

	detail::sm_set( aSmPrefix, _Env::lane(), aOp.identity() );
	barrier();

	detail::sm_set( tmp, _Env::lane(), aVal );
	barrier();

	T res = detail::BlockPrefixHelper<T,Op,_Env::SIMD,1,false,_Env::SYNC>::eval(
		aVal, aOp, _Env::lane(), tmp
	);
	barrier();

	return res;
}

template< class KSetup > template< typename T >
_CHAG_PP_DEV T Unit<KSetup>::prefix_total_get( volatile T* aSmPrefix )
{
	T ret = detail::sm_get( aSmPrefix, _Env::SIMD + _Env::SIMD/2 - 1 );
	barrier();

	return ret;
}

//--	Unit : seg. prefix	///{{{1///////////////////////////////////////////
namespace detail
{
	template< typename T, class Op, unsigned N, unsigned M, bool E, unsigned SYNC >
	struct SegBlockPrefixHelper
	{
		static _CHAG_PP_DEV T eval( const T& aV, const Op& aOp, bool aSegmentStart,
			unsigned aTID, volatile T* aSm, volatile unsigned* aSmSeg )
		{
			T temp;
			if( !aSegmentStart )
			{
				temp = aOp( smX_get_striped<N+N/2>( aSm, aTID-M ), aV ); 
				aSegmentStart = aSmSeg[aTID-M];
			}
			else
			{
				temp = aV;
			}
			detail::conditional_barrier<N,SYNC>();

			smX_set_striped<N+N/2>( aSm, aTID, temp );
			aSmSeg[aTID] = aSegmentStart;
			detail::conditional_barrier<N,SYNC>();

			return SegBlockPrefixHelper<T,Op,N,M*2,N==M*4,SYNC>::eval( 
				temp, aOp, aSegmentStart, aTID, aSm, aSmSeg
			);
		}
	};
	template< typename T, class Op, unsigned N, unsigned M, unsigned SYNC >
	struct SegBlockPrefixHelper<T,Op,N,M,true,SYNC>
	{
		static _CHAG_PP_DEV T eval( const T& aV, const Op& aOp, bool aSegmentStart,
			unsigned aTID, volatile T* aSm, volatile unsigned* aSmSeg )
		{
			T val;
			
			if( !aSegmentStart )
			{
				val = aOp( smX_get_striped<N+N/2>( aSm, aTID-M ), aV );
			}
			else
			{
				val = aV;
			}
			detail::conditional_barrier<N,SYNC>();

			smX_set_striped<N+N/2>( aSm, aTID, val );
			return val;
		}
	};
}

template< class KSetup > template< typename T, class Op >
_CHAG_PP_DEV T Unit<KSetup>::segmented_prefix_inclusive( 
	const T& aVal, const Op& aOp, bool aSegmentStart, volatile T* aSmPrefix, volatile unsigned* aSmSegments )
{
	typedef typename detail::PrimTypeOf<T>::type PT;
	enum { kBufferSize = _Env::SIMD + _Env::SIMD/2 };

	volatile T* tmp = (volatile T*)(((volatile PT*)aSmPrefix) + (_Env::SIMD/2));
	volatile unsigned* seg = aSmSegments + (_Env::SIMD/2);

	aSmSegments[_Env::lane()] = true;
	detail::smX_set_striped<kBufferSize>( aSmPrefix, _Env::lane(), aOp.identity() );
	barrier();

	seg[_Env::lane()] = aSegmentStart;
	detail::smX_set_striped<kBufferSize>( tmp, _Env::lane(), aVal );
	barrier();

	T res = detail::SegBlockPrefixHelper<T,Op,_Env::SIMD,1,false,_Env::SYNC>::eval(
		aVal, aOp, aSegmentStart, _Env::lane(), tmp, seg
	);
	barrier();

	return res;
}
template< class KSetup > template< typename T, class Op >
_CHAG_PP_DEV T Unit<KSetup>::segmented_prefix_inclusive( 
	const T& aVal, const T& aStartOffset, const Op& aOp, bool aSegmentStart, volatile T* aSmPrefix, volatile unsigned* aSmSegments )
{
	typedef typename detail::PrimTypeOf<T>::type PT;
	enum { kBufferSize = _Env::SIMD + _Env::SIMD/2 };

	volatile T* tmp = (volatile T*)(((volatile PT*)aSmPrefix) + (_Env::SIMD/2));
	volatile unsigned* seg = aSmSegments + (_Env::SIMD/2);

	T val = aVal;
	if( 0 == _Env::lane() && !aSegmentStart )
	{
		val = aOp( aStartOffset, aVal );
	}

	aSmSegments[_Env::lane()] = true;
	detail::smX_set_striped<kBufferSize>( aSmPrefix, _Env::lane(), aOp.identity() );
	barrier();

	seg[_Env::lane()] = aSegmentStart;
	detail::smX_set_striped<kBufferSize>( tmp, _Env::lane(), val );
	barrier();

	T res = detail::SegBlockPrefixHelper<T,Op,_Env::SIMD,1,false,_Env::SYNC>::eval(
		val, aOp, aSegmentStart, _Env::lane(), tmp, seg
	);
	barrier();

	return res;
}

//--	Unit : compact		///{{{1///////////////////////////////////////////
template< class KSetup > template< typename T >
_CHAG_PP_DEV SizeType Unit<KSetup>::compact( 
	T* aOut, const T& aVal, bool aKeep, volatile SizeType* aSmPrefix )
{
	T* dest = aOut + prefix<SizeType>( aKeep?1u:0u, op::Add<SizeType>(), aSmPrefix );
	SizeType count = prefix_total_get<SizeType>(aSmPrefix);

	if( aKeep )
	{
		*dest = aVal;
	}
	barrier();

	return count;
}
template< class KSetup > template< typename T >
_CHAG_PP_DEV SizeType Unit<KSetup>::compact(
	T* aOut, const T& aV0, const T& aV1, bool aKeep0, bool aKeep1,
	volatile SizeType* aSmPrefix )
{
	SizeType sum = 0;
	if( aKeep0 ) sum += 1;
	if( aKeep1 ) sum += 1;

	T* dest = aOut + prefix<SizeType>( sum, op::Add<SizeType>(), aSmPrefix );
	SizeType count = prefix_total_get<SizeType>(aSmPrefix);

	if( aKeep0 )
	{
		*dest = aV0;
		++dest;
	}
	if( aKeep1 )
	{
		*dest = aV1;
	}
	barrier();

	return count;
}

//--	Unit : split		///{{{1///////////////////////////////////////////
template< class KSetup > template< typename T >
_CHAG_PP_DEV SizeType Unit<KSetup>::split( 
	T* aOut, const T& aVal, bool aKeep, volatile SizeType* aSm )
{
	SizeType dest = prefix<SizeType>( aKeep ? 1u : 0u, op::Add<SizeType>(), aSm );
	SizeType count = prefix_total_get<SizeType>(aSm);

	if( !aKeep )
	{
		dest = count + _Env::lane() - dest;
	}
	aOut[dest] = aVal;
	barrier();

	return count;
}
template< class KSetup > template< typename T >
_CHAG_PP_DEV SizeType Unit<KSetup>::split( 
	T* aLeft, T* aRight, const T& aVal, bool aKeep, volatile SizeType* aSm )
{
	SizeType dest = prefix<SizeType>( aKeep ? 1u : 0u, op::Add<SizeType>(), aSm );
	SizeType count = prefix_total_get<SizeType>(aSm);

	T* of = aLeft;
	if( !aKeep )
	{
		of = aRight;
		dest = _Env::lane() - dest;
	}
	of[dest] = aVal;

	barrier();
	return count;
}


template< class KSetup > template< typename T >
_CHAG_PP_DEV SizeType Unit<KSetup>::split( 
	T* aOut, const T& aV0, const T& aV1, bool aK0, bool aK1, volatile SizeType* aSm )
{
	unsigned sum = 0;
	if( aK0 ) sum += 1;
	if( aK1 ) sum += 1;

	SizeType dest = prefix<SizeType>( sum, op::Add<SizeType>(), aSm );
	SizeType count = prefix_total_get<SizeType>(aSm);

	SizeType other = count + _Env::lane()*2 - dest;

	SizeType dest0 = dest;
	if( aK0 )
	{
		dest++;
	}
	else
	{
		dest0 = other;
		other++;
	}
	aOut[dest0] = aV0;

	SizeType dest1 = dest;
	if( !aK1 )
	{
		dest1 = other;
	}
	aOut[dest1] = aV1;

	barrier();
	return count;
}
template< class KSetup > template< typename T >
_CHAG_PP_DEV SizeType Unit<KSetup>::split( 
	T* aLeft, T* aRight, const T& aV0, const T& aV1, bool aK0, bool aK1, 
	volatile SizeType* aSm )
{
	unsigned sum = 0;
	if( aK0 ) sum += 1;
	if( aK1 ) sum += 1;

	SizeType dest = prefix<SizeType>( sum, op::Add<SizeType>(), aSm );
	SizeType count = prefix_total_get<SizeType>(aSm);

	SizeType other = _Env::lane()*2 - dest;

	T* of0;
	if( aK0 )
	{
		of0 = aLeft + dest;
		dest++;
	}
	else
	{
		of0 = aRight + other;
		other++;
	}
	*of0 = aV0;

	T* of1;
	if( aK1 )
	{
		of1 = aLeft + dest;
	}
	else
	{
		of1 = aRight + other;
	}
	*of1 = aV1;

	barrier();
	return count;
}

//--	Unit : split4		///{{{1///////////////////////////////////////////
template< class KSetup > template< typename T, class Pred0, class Pred1 >
_CHAG_PP_DEV SizeType Unit<KSetup>::split4( 
	T* aOut, const T& aVal, const Pred0& aPred0, const Pred1& aPred1, 
	volatile SizeType* aSm )
{
	bool pred0 = aPred0(aVal);
	bool pred1 = aPred1(aVal);

	SizeType idx = (pred0 ? 0u : 1u) + (pred1 ? 0u : 2u);
	SizeType inc = 1u << (idx << 3u); // WARN: works only if max/bucket < 255

	SizeType bdest = prefix<SizeType>( inc, op::Add<SizeType>(), aSm );
	SizeType count = prefix_total_get<SizeType>(aSm);

	count <<= 8u;
	count += count << 8u;
	count += count << 16u;			// WARN: works only if #total elems < 255

	SizeType dest = 0xFFu & ((count + bdest) >> (idx << 3u));
	aOut[dest] = aVal;
	barrier();

	return count;
}
template< class KSetup > template< typename T, class Pred0, class Pred1 >
_CHAG_PP_DEV SizeType Unit<KSetup>::split4( 
	T* aB0, T* aB1, T* aB2, T* aB3, const T& aVal, 
	const Pred0& aPred0, const Pred1& aPred1, volatile SizeType* aSm )
{
	bool pred0 = aPred0(aVal);
	bool pred1 = aPred1(aVal);

	SizeType idx = (pred0 ? 0u : 1u) + (pred1 ? 0u : 2u);
	SizeType inc = 1u << (idx << 3u); // WARN: works only if max/bucket < 255

	SizeType bdest = prefix<SizeType>( inc, op::Add<SizeType>(), aSm );
	SizeType count = prefix_total_get<SizeType>(aSm);
	
	SizeType dest = 0xFFu & (bdest >> (idx << 3u));
	T* of;

	switch( idx )
	{
		case 0: of = aB0; break;
		case 1: of = aB1; break;
		case 2: of = aB2; break;
		case 3: of = aB3; break;
	}

	of[dest] = aVal;
	barrier();

	return count;
}
template< class KSetup > template< typename T, class Pred0, class Pred1 >
_CHAG_PP_DEV SizeType Unit<KSetup>::split4( 
	T* aBase, volatile SizeType* aOffsets, const T& aVal, 
	const Pred0& aPred0, const Pred1& aPred1, volatile SizeType* aSm )
{
	bool pred0 = aPred0(aVal);
	bool pred1 = aPred1(aVal);

	SizeType idx = (pred0 ? 0u : 1u) + (pred1 ? 0u : 2u);
	SizeType inc = 1u << (idx << 3u); // WARN: works only if max/bucket < 255

	SizeType bdest = prefix<SizeType>( inc, op::Add<SizeType>(), aSm );
	SizeType count = prefix_total_get<SizeType>(aSm);
	
	SizeType dest = aOffsets[idx] + (0xFFu & (bdest >> (idx << 3u)));

	aBase[dest] = aVal;
	barrier();

	return count;
}

template< class KSetup > template< typename T, class Pred0, class Pred1 >
_CHAG_PP_DEV SizeType Unit<KSetup>::split4( 
	T* aBase, volatile SizeType* aOffsets, const T& aV0, const T& aV1, 
	const Pred0& aPred0, const Pred1& aPred1, volatile SizeType* aSm )
{
	/* eww :-( why do we have to do stuff like this?! */

	bool pred0a = aPred0(aV0);
	bool pred1a = aPred1(aV0);
	SizeType idxa = (pred0a ? 0u : 1u) + (pred1a ? 0u : 2u);

	bool pred0b = aPred0(aV1);
	bool pred1b = aPred1(aV1);
	SizeType idxb = (pred0b ? 0u : 1u) + (pred1b ? 0u : 2u);

	// WARN: overflow and such
	SizeType inc = (1u << (idxa << 3u)) + (1u << (idxb << 3u)); 

	SizeType bdest = prefix<SizeType>( inc, op::Add<SizeType>(), aSm );
	SizeType count = prefix_total_get<SizeType>(aSm);
	
	SizeType desta = aOffsets[idxa] + (0xFFu & (bdest >> (idxa << 3u)));
	SizeType destb = aOffsets[idxb] + (0xFFu & (bdest >> (idxb << 3u)));

	if( desta == destb )
		destb++;

	aBase[desta] = aV0;
	aBase[destb] = aV1;
	barrier();

	return count;
}

//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
CHAG_PP_LEAVE_NAMESPACE()
