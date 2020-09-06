/****************************************************************************/
/*!	\brief [chag::pp] Temporary buffer in global memory
 *
 * "HACK"
 *
 * We'd like to use device memory, e.g.
 *		__device__ type foo[BAR];
 * and let CUDA/nvcc deal with allocation of it, but in our case BAR depends
 * on the Job Setup. BAR _is_ known at compile time, however cuda does not
 * allow us to have a static __device__ member in any class.
 *
 * Meh. :-/
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

#ifndef _CHAG_PP_DETAIL_BUFFER_CUH_
#define _CHAG_PP_DETAIL_BUFFER_CUH_

//--//////////////////////////////////////////////////////////////////////////
//--	Include				///{{{1///////////////////////////////////////////

#include "../ext/common.cuh"

CHAG_PP_ENTER_NAMESPACE()
//--	Classes				///{{{1///////////////////////////////////////////
namespace detail
{
	class OffsetBuffer
	{
		public:
			static SizeType* count_buffer( unsigned aCount ) 
			{ 
				return assert_size(aCount).m_devCount; 
			}
			static SizeType* offset_buffer( unsigned aCount )
			{
				return assert_size(aCount).m_devOffset;
			}

		public:
			~OffsetBuffer() 
			{
				cudaFree( m_devCount );

				m_size = 0;
				m_devCount = m_devOffset = 0;
			}

		private:
			OffsetBuffer() : m_devCount(0), m_devOffset(0), m_size(0) {}

			SizeType m_size;
			SizeType* m_devCount;
			SizeType* m_devOffset;

			static OffsetBuffer& assert_size( SizeType aCount )
			{
				static OffsetBuffer instance;

				if( instance.m_size < aCount )
				{
					cudaFree( instance.m_devCount );

					SizeType size = aCount *sizeof(SizeType) * 2;
					cudaMalloc( (void**)&instance.m_devCount, size );
					
					instance.m_size = aCount;
					instance.m_devOffset = instance.m_devCount + aCount;
				}

				return instance;
			}
	};
}

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
#endif // _CHAG_PP_DETAIL_BUFFER_CUH_
