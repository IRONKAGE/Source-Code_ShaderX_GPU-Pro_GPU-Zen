/****************************************************************************/
/*!	\brief [chag::pp] range function implementations
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
//--	> default			///{{{1///////////////////////////////////////////
template< typename T, class KSetup, bool UsePairs > template< class Op >
_CHAG_PP_DEV T Range<T,KSetup,UsePairs>::reduce( 
	const T* aStart, const T* aEnd, const Op& aOp, volatile T* aSmReduce
)
{
	T partialSum = aOp.identity();;
	for( const T* elem = aStart + _Env::lane(); elem < aEnd; elem += _Env::SIMD )
	{
		partialSum = aOp( partialSum, *elem );
	}

	return _Unit::reduce( partialSum, aOp, aSmReduce );

}
template< typename T, class KSetup, bool UsePairs > template< class Pred >
_CHAG_PP_DEV SizeType Range<T,KSetup,UsePairs>::count( 
	const T* aStart, const T* aEnd, const Pred& aPred, volatile SizeType* aSmReduce
)
{
	SizeType partialCount = 0u;
	for( const T* elem = aStart + _Env::lane(); elem < aEnd; elem += _Env::SIMD )
	{
		if( aPred( *elem ) )
			++partialCount;
	}

	return _Unit::reduce( partialCount, op::Add<SizeType>(), aSmReduce );
}

template< typename T, class KSetup, bool UsePairs > template< class Op >
_CHAG_PP_DEV T Range<T,KSetup,UsePairs>::prefix( 
	const T* aStart, const T* aEnd, T* aOut, T aOffset, const Op& aOp, 
	volatile T* aSmPrefix )
{
	T offset = aOffset;
	T* out = aOut + _Env::lane();

	for( const T* elem = aStart + _Env::lane(); elem < aEnd; elem += _Env::SIMD )
	{
		*out = aOp( offset, _Unit::prefix( *elem, aOp, aSmPrefix ) );
		offset = aOp( offset, _Unit::prefix_total_get( aSmPrefix ) );

		out += _Env::SIMD;
	}

	return offset;
}
template< typename T, class KSetup, bool UsePairs > template< class Op >
_CHAG_PP_DEV T Range<T,KSetup,UsePairs>::prefix_inclusive( 
	const T* aStart, const T* aEnd, T* aOut, T aOffset, const Op& aOp, 
	volatile T* aSmPrefix )
{
	T offset = aOffset;
	T* out = aOut + _Env::lane();

	for( const T* elem = aStart + _Env::lane(); elem < aEnd; elem += _Env::SIMD )
	{
		*out = aOp( offset, _Unit::prefix_inclusive( *elem, aOp, aSmPrefix ) );
		offset = aOp( offset, _Unit::prefix_total_get( aSmPrefix ) );

		out += _Env::SIMD;
	}

	return offset;
}


template< typename T, class KSetup, bool UsePairs > template< class Pred >
_CHAG_PP_DEV SizeType Range<T,KSetup,UsePairs>::compact_scatter( 
	const T* aStart, const T* aEnd, T* aOut, const Pred& aPred, 
	volatile SizeType* aSmPrefix )
{
	T* out = aOut;
	for( const T* elem = aStart + _Env::lane(); elem < aEnd; elem += _Env::SIMD )
	{
		T e = *elem;
		out += _Unit::compact( out, e, aPred(e), aSmPrefix );
	}

	return out - aOut;
}
template< typename T, class KSetup, bool UsePairs > template< class Pred >
_CHAG_PP_DEV SizeType Range<T,KSetup,UsePairs>::compact_stage(
	const T* aStart, const T* aEnd, T* aOut, const Pred& aPred, 
	volatile T* aSmStage, volatile SizeType* aSmPrefix )
{
	T* out = aOut + _Env::lane();
	T* src = (T*)aSmStage + _Env::lane();

	for( const T* elem = aStart + _Env::lane(); elem < aEnd; elem += _Env::SIMD )
	{
		T e = *elem;
		SizeType count = _Unit::compact( (T*)aSmStage, e, aPred(e), aSmPrefix );
		if( _Env::lane() < count )
		{
			*out = *src;
		}

		out += count;
	}

	return out - aOut;
}


template< typename T, class KSetup, bool UsePairs > template< class Pred >
_CHAG_PP_DEV SizeType Range<T,KSetup,UsePairs>::split_scatter(
	const T* aStart, const T* aEnd, T* aLeft, T* aRight, const Pred& aPred, 
	volatile SizeType* aSmPrefix )
{
	T* left = aLeft;
	T* right = aRight;

	for( const T* elem = aStart + _Env::lane(); elem < aEnd; elem += _Env::SIMD )
	{
		T e = *elem;
		SizeType count = _Unit::split( left, right, e, aPred(e), aSmPrefix );
		SizeType workaround = _Env::SIMD - count; // see far below

		left += count;
		right += workaround;
	}

	return left - aLeft - _Env::lane();
}
template< typename T, class KSetup, bool UsePairs > template< class Pred >
_CHAG_PP_DEV SizeType Range<T,KSetup,UsePairs>::split_stage(
	const T* aStart, const T* aEnd, T* aLeft, T* aRight, const Pred& aPred, 
	volatile T* aSmStage, volatile SizeType* aSmPrefix )
{
	T* left = aLeft + _Env::lane();
	T* right = aRight + _Env::lane();

	for( const T* elem = aStart + _Env::lane(); elem < aEnd; elem += _Env::SIMD )
	{
		T e = *elem;
		SizeType count = _Unit::split( (T*)aSmStage, e, aPred(e), aSmPrefix );

		T* of = left;
		if( _Env::lane() >= count )
			of = right - count;

		*of = ((T*)aSmStage)[_Env::lane()];
		_Unit::barrier();
		SizeType workaround = _Env::SIMD - count; // see far below

		left += count;
		right += workaround;
	}

	return left - aLeft - _Env::lane();
}

// Special: 4-way things
template< typename T, class KSetup, bool UsePairs > template< class Pred0, class Pred1 >
_CHAG_PP_DEV Range<T,KSetup,UsePairs>::SizeType4 Range<T,KSetup,UsePairs>::count4(
	const T* aStart, const T* aEnd, const Pred0& aPred0, const Pred1& aPred1,
	volatile SizeType* aSmReduce )
{
	SizeType pc0 = 0, pc1 = 0, pc2 = 0, pc3 = 0;
	for( const T* elem = aStart + _Env::lane(); elem < aEnd; elem += _Env::SIMD )
	{
		T e = *elem;

		bool pred0 = aPred0(e);
		bool pred1 = aPred1(e);

		/* TODO: can we do this nicer? */
		if( pred0 )
		{
			if( pred1 )
				++pc0;
			else
				++pc2;
		}
		else
		{
			if( pred1 )
				++pc1;
			else
				++pc3;
		}
	}

	SizeType4 sum;
	sum.x = _Unit::reduce( pc0, op::Add<SizeType>(), aSmReduce );
	sum.y = _Unit::reduce( pc1, op::Add<SizeType>(), aSmReduce );
	sum.z = _Unit::reduce( pc2, op::Add<SizeType>(), aSmReduce );
	sum.w = _Unit::reduce( pc3, op::Add<SizeType>(), aSmReduce );
	return sum;
}
template< typename T, class KSetup, bool UsePairs > template< class Pred0, class Pred1 >
_CHAG_PP_DEV Range<T,KSetup,UsePairs>::SizeType4 Range<T,KSetup,UsePairs>::count4(
	const T* aStart, const T* aEnd, const Pred0& aPred0, const Pred1& aPred1,
	volatile SizeType* aSmCount, volatile SizeType* aSmReduce )
{
	aSmCount[0*_Env::SIMD + _Env::lane()] = 0;
	aSmCount[1*_Env::SIMD + _Env::lane()] = 0;
	aSmCount[2*_Env::SIMD + _Env::lane()] = 0;
	aSmCount[3*_Env::SIMD + _Env::lane()] = 0;

	for( const T* elem = aStart + _Env::lane(); elem < aEnd; elem += _Env::SIMD )
	{
		T e = *elem;

		bool pred0 = aPred0(e);
		bool pred1 = aPred1(e);

		SizeType idx = (pred0 ? 0u : _Env::SIMD) + (pred1 ? 0u : 2*_Env::SIMD);
		++aSmCount[idx + _Env::lane()];
	}

	SizeType pc0, pc1, pc2, pc3;
	pc0 = aSmCount[0*_Env::SIMD + _Env::lane()];
	pc1 = aSmCount[1*_Env::SIMD + _Env::lane()];
	pc2 = aSmCount[2*_Env::SIMD + _Env::lane()];
	pc3 = aSmCount[3*_Env::SIMD + _Env::lane()];
	_Unit::barrier();

	SizeType4 sum;
	sum.x = _Unit::reduce( pc0, op::Add<SizeType>(), aSmReduce );
	sum.y = _Unit::reduce( pc1, op::Add<SizeType>(), aSmReduce );
	sum.z = _Unit::reduce( pc2, op::Add<SizeType>(), aSmReduce );
	sum.w = _Unit::reduce( pc3, op::Add<SizeType>(), aSmReduce );
	return sum;
}


template< typename T, class KSetup, bool UsePairs > template< class Pred0, class Pred1 >
_CHAG_PP_DEV void Range<T,KSetup,UsePairs>::split4_scatter(
	const T* aStart, const T* aEnd, T* aBase, volatile SizeType* aOffsets,
	const Pred0& aPred0, const Pred1& aPred1, 
	volatile SizeType* aSmPrefix )
{
	for( const T* elem = aStart + _Env::lane(); elem < aEnd; elem += _Env::SIMD )
	{
		SizeType counts = _Unit::split4( 
			aBase, aOffsets, *elem, aPred0, aPred1, aSmPrefix 
		);

		// update offsets
		if( _Env::lane() < 4 )
		{
			aOffsets[_Env::lane()] += 0xFFu & (counts >> (_Env::lane() << 3u));
		}
		_Unit::barrier();
	}
}


//--	> pairs				///{{{1///////////////////////////////////////////
template< typename T, class KSetup > 
struct Range<T,KSetup,true>
{
	template< class Op > static _CHAG_PP_DEV T reduce( 
		const T* aStart, const T* aEnd, const Op& aOp, volatile T* aSmReduce
	)
	{
		_Tp partialSum = { aOp.identity(), aOp.identity() };
		for( const _Tp* elem = _Tpp(aStart) + _Env::lane(); elem < _Tpp(aEnd); 
			elem += _Env::SIMD )
		{
			_Tp e = *elem;

			partialSum.x = aOp( partialSum.x, e.x ); 
			partialSum.y = aOp( partialSum.y, e.y ); 
		}

		T sum = aOp( partialSum.x, partialSum.y );
		return _Unit::reduce( sum, aOp, aSmReduce );
	}
	template< class Pred > static _CHAG_PP_DEV SizeType count( 
		const T* aStart, const T* aEnd, const Pred& aPred, volatile SizeType* aSmReduce
	)
	{
		SizeType partialCountA = 0u, partialCountB = 0u;
		for( const _Tp* elem = _Tpp(aStart) + _Env::lane(); elem < _Tpp(aEnd); 
			elem += _Env::SIMD )
		{
			_Tp e = *elem;

			if( aPred( e.x ) )
				++partialCountA;
			if( aPred( e.y ) )
				++partialCountB;
		}

		SizeType sum = partialCountA + partialCountB;
		return _Unit::reduce( sum, op::Add<SizeType>(), aSmReduce );
	}

	template< class Op > static _CHAG_PP_DEV T prefix(
		const T* aStart, const T* aEnd, T* aOut, T aOffset, const Op& aOp,
		volatile T* aSmPrefix )
	{
		T offset = aOffset;
		_Tp* out = _Tpp(aOut) + _Env::lane();

		for( const _Tp* elem = _Tpp(aStart) + _Env::lane(); elem < _Tpp(aEnd); 
			elem += _Env::SIMD )
		{
			_Tp e = *elem;
			T sum = aOp( e.x, e.y );
			T scn = aOp( offset, _Unit::prefix( sum, aOp, aSmPrefix ) );

			_Tp o = { scn, aOp(scn,e.x) };
			*out = o;

			offset = aOp( offset, _Unit::prefix_total_get( aSmPrefix ) );
			out += _Env::SIMD;
		}

		return offset;
	}
	template< class Op > static _CHAG_PP_DEV T prefix_inclusive(
		const T* aStart, const T* aEnd, T* aOut, T aOffset, const Op& aOp,
		volatile T* aSmPrefix )
	{
		T offset = aOffset;
		_Tp* out = _Tpp(aOut) + _Env::lane();

		for( const _Tp* elem = _Tpp(aStart) + _Env::lane(); elem < _Tpp(aEnd); 
			elem += _Env::SIMD )
		{
			_Tp e = *elem;
			T sum = aOp( e.x, e.y );
			T scn = aOp( offset, _Unit::prefix( sum, aOp, aSmPrefix ) );

			T a = aOp( scn, e.x );
			offset = aOp( offset, _Unit::prefix_total_get( aSmPrefix ) );

			T b = aOp( a, e.y );

			_Tp o = { a, b };
			*out = o;
			out += _Env::SIMD;
		}

		return offset;
	}


	template< class Pred > static _CHAG_PP_DEV SizeType compact_scatter(
		const T* aStart, const T* aEnd, T* aOut, const Pred& aPred, 
		volatile SizeType* aSmPrefix )
	{
		T* out = aOut;
		for( const _Tp* elem = _Tpp(aStart) + _Env::lane(); elem < _Tpp(aEnd); 
			elem += _Env::SIMD )
		{
			_Tp e = *elem;
			out += _Unit::compact( out, e.x, e.y, aPred(e.x), aPred(e.y), aSmPrefix );
		}

		return out - aOut;
	}
	template< class Pred > static _CHAG_PP_DEV SizeType compact_stage(
		const T* aStart, const T* aEnd, T* aOut, const Pred& aPred, 
		volatile T* aSmStage, volatile SizeType* aSmPrefix )
	{
		const SizeType farLane = _Env::lane() + _Env::SIMD;

		T* out = aOut;
		for( const _Tp* elem = _Tpp(aStart) + _Env::lane(); elem < _Tpp(aEnd); 
			elem += _Env::SIMD )
		{
			_Tp e = *elem;
			SizeType count = _Unit::compact( (T*)aSmStage, e.x, e.y, 
				aPred(e.x), aPred(e.y), aSmPrefix 
			);

			if( _Env::lane() < count )
				out[_Env::lane()] = aSmStage[_Env::lane()];

			if( farLane < count )
				out[farLane] = aSmStage[farLane];

			out += count;
		}

		return out - aOut;
	}

	template< class Pred > static _CHAG_PP_DEV SizeType split_scatter(
		const T* aStart, const T* aEnd, T* aLeft,  T* aRight, const Pred& aPred, 
		volatile SizeType* aSmPrefix )
	{
		T* left = aLeft;
		T* right = aRight;

		for( const _Tp* elem = _Tpp(aStart) + _Env::lane(); elem < _Tpp(aEnd); 
			elem += _Env::SIMD )
		{
			_Tp e = *elem;
			SizeType count = _Unit::split( left, right, e.x, e.y, 
				aPred(e.x), aPred(e.y), aSmPrefix );

			SizeType workaround = _Env::SIMD*2-count; 
			/* The line above is for nvcc release 3.1, V0.2.1221, which 
			 * otherwise messes up stuff (on the Fermi?): 
			 *
			 *  right => 0x5224200
			 *  _Env::SIMD*2-count => 35
			 *  right += _ENV::SIMD*2-count => 0x40522428c
			 *
			 * *insert sound of huge explosion here*
			 *
			 * (Original Code was: 
			 * 	right += _Env::SIMD*2-count;
			 * )
			 */

			left += count;
			right += workaround;
		}

		return left - aLeft - _Env::lane();
	}
	template< class Pred > static _CHAG_PP_DEV SizeType split_stage(
		const T* aStart, const T* aEnd, T* aLeft,  T* aRight, const Pred& aPred, 
		volatile T* aSmStage, volatile SizeType* aSmPrefix )
	{
		T* left = aLeft + _Env::lane();
		T* right = aRight + _Env::lane();

		for( const _Tp* elem = _Tpp(aStart) + _Env::lane(); elem < _Tpp(aEnd); 
			elem += _Env::SIMD )
		{
			_Tp e = *elem;
			SizeType count = _Unit::split( (T*)aSmStage, e.x, e.y, 
				aPred(e.x), aPred(e.y), aSmPrefix );

			// flush first 32
			T* of = left;
			if( _Env::lane() >= count )
				of = right - count;

			*of = aSmStage[_Env::lane()];

			// flush second 32
			of = left + _Env::SIMD;
			if( _Env::lane()+_Env::SIMD >= count )
				of = right + _Env::SIMD - count;

			*of = aSmStage[_Env::lane()+_Env::SIMD];

			_Unit::barrier();
			SizeType workaround = _Env::SIMD*2-count; // see above somewhere

			left += count;
			right += workaround;
		}

		return left - aLeft - _Env::lane();
	}

	// Special: 4-way things
	typedef uint4 SizeType4;

	template< class Pred0, class Pred1 > static _CHAG_PP_DEV SizeType4 count4(
		const T* aStart, const T* aEnd, const Pred0& aPred0, const Pred1& aPred1,
		volatile SizeType* aSmReduce )
	{
		SizeType pc0 = 0, pc1 = 0, pc2 = 0, pc3 = 0;
		for( const _Tp* elem = _Tpp(aStart) + _Env::lane(); elem < _Tpp(aEnd); 
			elem += _Env::SIMD )
		{
			_Tp e = *elem;

			{
				bool pred0 = aPred0(e.x);
				bool pred1 = aPred1(e.x);

				/* TODO Can we do this nicer? */
				if( pred0 )
				{
					if( pred1 )
						++pc0;
					else
						++pc2;
				}
				else
				{
					if( pred1 )
						++pc1;
					else
						++pc3;
				}
			}
			{
				bool pred0 = aPred0(e.y);
				bool pred1 = aPred1(e.y);

				if( pred0 )
				{
					if( pred1 )
						++pc0;
					else
						++pc2;
				}
				else
				{
					if( pred1 )
						++pc1;
					else
						++pc3;
				}
			}

		}

		SizeType4 sum;
		sum.x = _Unit::reduce( pc0, op::Add<SizeType>(), aSmReduce );
		sum.y = _Unit::reduce( pc1, op::Add<SizeType>(), aSmReduce );
		sum.z = _Unit::reduce( pc2, op::Add<SizeType>(), aSmReduce );
		sum.w = _Unit::reduce( pc3, op::Add<SizeType>(), aSmReduce );
		return sum;
	}
	template< class Pred0, class Pred1 > static _CHAG_PP_DEV SizeType4 count4(
		const T* aStart, const T* aEnd, const Pred0& aPred0, const Pred1& aPred1,
		volatile SizeType* aSmCount, volatile SizeType* aSmReduce )
	{
		aSmCount[0*_Env::SIMD + _Env::lane()] = 0;
		aSmCount[1*_Env::SIMD + _Env::lane()] = 0;
		aSmCount[2*_Env::SIMD + _Env::lane()] = 0;
		aSmCount[3*_Env::SIMD + _Env::lane()] = 0;

		for( const _Tp* elem = _Tpp(aStart) + _Env::lane(); elem < _Tpp(aEnd); 
			elem += _Env::SIMD )
		{
			_Tp e = *elem;

			{
				bool pred0 = aPred0(e.x);
				bool pred1 = aPred1(e.x);

				SizeType idx = (pred0 ? 0u : _Env::SIMD) + (pred1 ? 0u : 2*_Env::SIMD);
				++aSmCount[idx + _Env::lane()];
			}
			{
				bool pred0 = aPred0(e.y);
				bool pred1 = aPred1(e.y);

				SizeType idx = (pred0 ? 0u : _Env::SIMD) + (pred1 ? 0u : 2*_Env::SIMD);
				++aSmCount[idx + _Env::lane()];
			}

		}

		SizeType pc0, pc1, pc2, pc3;
		pc0 = aSmCount[0*_Env::SIMD + _Env::lane()];
		pc1 = aSmCount[1*_Env::SIMD + _Env::lane()];
		pc2 = aSmCount[2*_Env::SIMD + _Env::lane()];
		pc3 = aSmCount[3*_Env::SIMD + _Env::lane()];
		_Unit::barrier();

		SizeType4 sum;
		sum.x = _Unit::reduce( pc0, op::Add<SizeType>(), aSmReduce );
		sum.y = _Unit::reduce( pc1, op::Add<SizeType>(), aSmReduce );
		sum.z = _Unit::reduce( pc2, op::Add<SizeType>(), aSmReduce );
		sum.w = _Unit::reduce( pc3, op::Add<SizeType>(), aSmReduce );
		return sum;
	}


	template< class Pred0, class Pred1 > static _CHAG_PP_DEV void split4_scatter(
		const T* aStart, const T* aEnd, T* aBase, volatile SizeType* aOffsets,
		const Pred0& aPred0, const Pred1& aPred1, volatile SizeType* aSmPrefix )
	{
		for( const _Tp* elem = _Tpp(aStart) + _Env::lane(); elem < _Tpp(aEnd); 
			elem += _Env::SIMD )
		{
			_Tp e = *elem;

			SizeType counts = _Unit::split4( 
				aBase, aOffsets, e.x, e.y, aPred0, aPred1, aSmPrefix 
			);

			// update offsets
			if( _Env::lane() < 4 )
			{
				aOffsets[_Env::lane()] += 0xFFu & (counts >> (_Env::lane() << 3u));
			}
			_Unit::barrier();
		}
	}

	/* Mostly copy-pasta. Is this really needed?! */
	protected:
		typedef typename KSetup::Env _Env;
		typedef Unit<KSetup> _Unit;

		typedef typename detail::PairTypeOf<T>::type _Tp;
		typedef typename detail::PairTypeOf<T>::type* _Tpp;

	public:
		enum { MEMORY_ELEMENTS_STAGE = _Env::SIMD * 2 };

		enum { MEMORY_ELEMENTS_REDUCE = _Unit::MEMORY_ELEMENTS_REDUCE };
		enum { MEMORY_ELEMENTS_PREFIX = _Unit::MEMORY_ELEMENTS_PREFIX };

};

//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
CHAG_PP_LEAVE_NAMESPACE()
