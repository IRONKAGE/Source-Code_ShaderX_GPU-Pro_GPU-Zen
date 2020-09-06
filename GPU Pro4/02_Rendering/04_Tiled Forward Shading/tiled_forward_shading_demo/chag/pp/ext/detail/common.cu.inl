/****************************************************************************/
/*!	\brief [chag::pp] common function implementation
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
//--	d :: cond_barrier	///{{{1///////////////////////////////////////////
namespace detail
{
	template< bool B > struct SyncIfHelper
	{
		static _CHAG_PP_DEV void sync();
	};

	template<> _CHAG_PP_DEV void SyncIfHelper<true>::sync()
	{
		__syncthreads();
	}
	template<> _CHAG_PP_DEV void SyncIfHelper<false>::sync()
	{}

	template< unsigned N, unsigned SIMD >
	_CHAG_PP_DEV void conditional_barrier()
	{
		detail::SyncIfHelper< (N>SIMD) >::sync();
	}
}

//--	d :: nextPow2		///{{{1///////////////////////////////////////////
namespace detail
{
	template< unsigned N > 
	struct NextPowerOf2
	{
		private:
			enum { t1 = N-1 };
			enum { t2 = t1 | (t1 >> 1) };
			enum { t3 = t2 | (t2 >> 2) };
			enum { t4 = t3 | (t3 >> 4) };
			enum { t5 = t4 | (t4 >> 8) };
			enum { t6 = t5 | (t5 >> 16) };

		public:
			enum { value = t6 + 1 };
	};
}

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
