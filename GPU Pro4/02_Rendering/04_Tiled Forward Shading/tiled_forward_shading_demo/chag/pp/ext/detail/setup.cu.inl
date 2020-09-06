/****************************************************************************/
/*!	\brief [chag::pp] setup implementation
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
//--	JobSetup			///{{{1///////////////////////////////////////////
template< typename T, class KSetup, bool UsePairs >
void JobSetupStatic<T,KSetup,UsePairs>::setup( SizeType aNumElements )
{
	m_kernelParam.chunksBase = (aNumElements / CHUNK_SIZE) / JOB_COUNT;
	m_kernelParam.chunksExtra = (aNumElements / CHUNK_SIZE) % JOB_COUNT;

	m_kernelParam.auxElements = aNumElements % CHUNK_SIZE;
	m_kernelParam.auxStart = aNumElements - m_kernelParam.auxElements;
}
template< typename T, class KSetup, bool UsePairs >
const JobSetupStatic<T,KSetup,UsePairs>::KernelParam&
	JobSetupStatic<T,KSetup,UsePairs>::kernelParam() const
{
	return m_kernelParam;
}

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
