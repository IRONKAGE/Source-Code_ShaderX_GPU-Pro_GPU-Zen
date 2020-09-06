/****************************************************************************/
/*!	\brief [chag::pp] Kernel & Job Setup
 *
 * Defines classes 
 * 		- \c KernelSetup -- Kernel setup
 *		- \c JobSetupStatic -- Static job distribution
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

#ifndef _CHAG_PP_EXT_SETUP_CUH_
#define _CHAG_PP_EXT_SETUP_CUH_

//--//////////////////////////////////////////////////////////////////////////
//--	Include				///{{{1///////////////////////////////////////////

#include "common.cuh"
#include "traits.cuh"

#include "../detail/nta.hpp"

CHAG_PP_ENTER_NAMESPACE()
//--	Kernel Setups		///{{{1///////////////////////////////////////////
/*! \brief Kernel setup - Unit = Warp
 *
 * Kernel setup where one warp is used as an idependent unit.
 */
template<
	unsigned BlockCount = CHAG_PP_DEFAULT_BLOCKS,
	unsigned ThreadCount = CHAG_PP_DEFAULT_THREADS,
	unsigned WarpSize = 32
> struct KernelSetupWarp
{
	struct Env
	{
		static const unsigned BLOCKS = BlockCount;
		static const unsigned THREADS = ThreadCount;

		static const unsigned SIMD = WarpSize;
		static const unsigned SYNC = WarpSize;

		static const unsigned CORES = THREADS/SIMD;
		static const unsigned UNITS = BLOCKS * CORES;
		
		static _CHAG_PP_DEV SizeType lane() { return threadIdx.x; }
		static _CHAG_PP_DEV SizeType core() { return threadIdx.y; }
		static _CHAG_PP_DEV SizeType unit() { return threadIdx.y+blockIdx.x*CORES; }
	};

	static inline dim3 layout_cuda_blocks() { 
		return dim3(Env::UNITS/Env::CORES,1,1); 
	}
	static inline dim3 layout_cuda_threads() { 
		return dim3(Env::SIMD,Env::CORES,1); 
	}
};

/* \brief Kernel setup - Unit = Block
 *
 * Kernel setup where one block is used as an independent unit.
 */
template<
	unsigned BlockCount,
	unsigned ThreadCount,
	unsigned WarpSize = 32
> struct KernelSetupBlock
{
	struct Env
	{
		static const unsigned BLOCKS = BlockCount;
		static const unsigned THREADS = ThreadCount;

		static const unsigned SIMD = ThreadCount;
		static const unsigned SYNC = WarpSize;

		static const unsigned CORES = 1;
		static const unsigned UNITS = BLOCKS;
		
		static _CHAG_PP_DEV SizeType lane() { return threadIdx.x; }
		static _CHAG_PP_DEV SizeType core() { return 0; }
		static _CHAG_PP_DEV SizeType unit() { return blockIdx.x; }
	};

	static inline dim3 layout_cuda_blocks() { 
		return dim3(Env::UNITS/Env::CORES,1,1); 
	}
	static inline dim3 layout_cuda_threads() { 
		return dim3(Env::SIMD,Env::CORES,1); 
	}
};

//--	Job Setups			///{{{1///////////////////////////////////////////
/* \brief Static load balancing
 *
 * Perform static load balancing for a specific kernel setup, \a KSetup, and
 * a specific input type.
 *
 * TODO: docs
 */
template< 
	typename T, 
	class KSetup,
	bool UsePairs = detail::HasPair<T>::value
> struct JobSetupStatic : public KSetup
{
	struct KernelParam
	{
		SizeType chunksBase;
		SizeType chunksExtra;

		SizeType auxStart;
		SizeType auxElements;
	};
	struct UnitSetup
	{
		SizeType id;
		SizeType begin, end;
	};

#if 0
#if 0
	struct Env : public KSetup::Env
	{
		enum { PAIRS_ENABLED = UsePairs };
	};
//	Bug: nvcc creates invalid code when we use Env::*; so we have to make sure
//	those constants actually exist in the Env Struct
#else
	struct Env : public KSetup::Env /* HACK */
	{
		enum { SIMD = KSetup::Env::SIMD };
		enum { SYNC = KSetup::Env::SYNC };

		enum { CORES = KSetup::Env::CORES };
		enum { UNITS = KSetup::Env::UNITS };

		enum { PAIRS_ENABLED = UsePairs };
	};
#endif
#endif

	inline void setup( SizeType aNumElemens );
	inline const KernelParam& kernelParam() const;

	private:
		enum { CHUNK_MULT = UsePairs ? 2u : 1u };
		enum { JOB_MULT = 1 /*TODO: job multiplier*/ };

	public:
		enum { CHUNK_SIZE = CHUNK_MULT * KSetup::Env::SIMD  };
		enum { JOB_COUNT = JOB_MULT * KSetup::Env::UNITS };
		enum { JOB_BUFFER_SIZE = detail::NextPowerOf2<JOB_COUNT+1>::value };

	private:
		KernelParam m_kernelParam;

	public:
		static _CHAG_PP_DEV void perform_unit_setup_( 
			UnitSetup& aJobSpec, 
			unsigned aJobId, 
			const KernelParam& aKS 
		)
		{
			aJobSpec.id = aJobId;
			aJobSpec.begin = CHUNK_SIZE * (aKS.chunksBase * aJobId 
				+ ::min( aJobId, aKS.chunksExtra ));

			unsigned count = CHUNK_SIZE * (aKS.chunksBase 
				+ (aKS.chunksExtra > aJobId ? 1 : 0));

			aJobSpec.end = aJobSpec.begin + count;
		}
};

//--	Macro Magic			///{{{1///////////////////////////////////////////

#define CHAG_PP_KERNEL_SETUP_none
#define CHAG_PP_KERNEL_SETUP_persist( arg )							\
	__shared__ struct _chag_pp_sharedPersist { 						\
		arg 														\
	} _chag_pp_shared_persist[_Env::CORES];							\
	_chag_pp_sharedPersist& _sm_persist 							\
		= _chag_pp_shared_persist[_Env::core()];					\
	/*END*/
#define CHAG_PP_KERNEL_SETUP_temp( arg )							\
	__shared__ union _chag_pp_sharedTemp { 							\
		arg 														\
	} _chag_pp_shared_temp[_Env::CORES];							\
	_chag_pp_sharedTemp& _sm_temp 									\
		= _chag_pp_shared_temp[_Env::core()];						\
	/*END*/
#define CHAG_PP_KERNEL_SETUP_jobs( param, input )					\
	__shared__ typename _chag_pp_KSetup::UnitSetup 					\
		_chag_pp_unit_setup[_Env::CORES];							\
	if( _Env::core() == 0 && _Env::lane() < _Env::CORES ) {			\
		_chag_pp_KSetup::perform_unit_setup_(  						\
			_chag_pp_unit_setup[_Env::lane()], 						\
			_Env::unit() + _Env::lane(), param						\
		);															\
	}																\
	detail::conditional_barrier<_Env::CORES,1>();					\
	detail::conditional_barrier<_Env::SIMD,_Env::SYNC>();			\
	const typename _chag_pp_KSetup::UnitSetup& _job = 				\
		_chag_pp_unit_setup[_Env::core()];							\
	/*END*/
#define CHAG_PP_KERNEL_SETUP_init( tasks )							\
	if( _Env::core() == 0 && _Env::lane() < _Env::CORES ) {			\
		_chag_pp_sharedPersist& persist = 							\
			_chag_pp_shared_persist[_Env::lane()];					\
																	\
		tasks														\
	}																\
	detail::conditional_barrier<_Env::CORES,1>();					\
	detail::conditional_barrier<_Env::SIMD,_Env::SYNC>();			\
	/*END*/



#define CHAG_PP_KERNEL_SETUP( KSetup, a, b, c, d )					\
	typedef KSetup _chag_pp_KSetup;									\
	typedef typename KSetup::Env _Env;								\
																	\
	CHAG_PP_KERNEL_SETUP_##a 										\
	CHAG_PP_KERNEL_SETUP_##b										\
	CHAG_PP_KERNEL_SETUP_##c										\
	CHAG_PP_KERNEL_SETUP_##d										\
	/*END*/

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#include "detail/setup.cu.inl"
#endif // _CHAG_PP_EXT_SETUP_CUH_
