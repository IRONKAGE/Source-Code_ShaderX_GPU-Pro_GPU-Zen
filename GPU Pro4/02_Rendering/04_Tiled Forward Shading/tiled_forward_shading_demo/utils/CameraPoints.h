/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson
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
#ifndef _CameraPoints_h_
#define _CameraPoints_h_

#include <linmath/float3.h>
#include <string>
#include <vector>

namespace chag
{

/**
 */
class CameraPoints
{
public:
	CameraPoints();

	struct Snap
	{
		float3 fwd;
		float3 up;
		float3 pos;
	  float moveVel;
		float moveVelMult;
		bool alternativeControls;
	};

	bool load(const std::string &fileName);
	bool save(const std::string &fileName = std::string());
  /**
   * Uses the keys 
   *  
   *  returns true if key was handled.
   */
  bool handleKeyInput(uint8_t keyCode, bool down);
	/**
	 */
	Snap nextSnap();
	/**
	 */
	Snap getCurrentSnap() const { return m_snaps[m_currentSnap]; }
	/**
	 */
	void addSnap(const Snap &snap);
	/**
	 */
	void removeSnap(uint32_t at = ~0U);
	/**
	 */
	bool empty() const { return m_snaps.empty(); }

protected:
	std::vector<Snap> m_snaps;
	uint32_t m_currentSnap;
	std::string m_fileName;
};



}; // namespace chag


#endif // _CameraPoints_h_
