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
#include "CameraPoints.h"
#include "PlatformCompat.h"

#include <ctype.h>
#include <stdio.h>

namespace chag
{



CameraPoints::CameraPoints()
{
	m_currentSnap = 0;
}


bool CameraPoints::load(const std::string &fileName)
{
	m_fileName = fileName;
	if (FILE *f = fopen(m_fileName.c_str(), "r"))
	{
		fscanf(f, "current = %d\n", &m_currentSnap);
		char buffo[12];
		Snap s;
		while(fscanf(f, "{ %f, %f, %f }, { %f, %f, %f }, { %f, %f, %f }, %f, %f, %s\n", &s.fwd.x, &s.fwd.y, &s.fwd.z, &s.up.x, &s.up.y, &s.up.z, &s.pos.x, &s.pos.y, &s.pos.z, &s.moveVel, &s.moveVelMult, buffo) == 12)
		{
			s.alternativeControls = std::string(buffo) == "true";
			m_snaps.push_back(s);
		}
		m_currentSnap = m_snaps.empty() ? 0U : m_currentSnap % uint32_t(m_snaps.size());
		fclose(f);
	}
	return false;}



bool CameraPoints::save(const std::string &fileName)
{
	if (!fileName.empty())
	{
		m_fileName = fileName;
	}
	if (FILE *f = fopen(m_fileName.c_str(), "w"))
	{
		fprintf(f, "current = %d\n", m_currentSnap);
		for (size_t i = 0; i < m_snaps.size(); ++i)
		{
			Snap s = m_snaps[i];
			fprintf(f, "{ %f, %f, %f }, { %f, %f, %f }, { %f, %f, %f }, %f, %f, %s\n", s.fwd.x, s.fwd.y, s.fwd.z, s.up.x, s.up.y, s.up.z, s.pos.x, s.pos.y, s.pos.z, s.moveVel, s.moveVelMult, s.alternativeControls ? "true" : "false");
		}
		fclose(f);
	}
	return false;
}



bool CameraPoints::handleKeyInput(uint8_t keyCode, bool down)
{
	return false;
}



CameraPoints::Snap CameraPoints::nextSnap()
{
	m_currentSnap = (m_currentSnap + 1) % uint32_t(m_snaps.size());
	return m_snaps[m_currentSnap];
}



void CameraPoints::addSnap(const Snap &snap)
{
	m_snaps.push_back(snap);
}



void CameraPoints::removeSnap(uint32_t at)
{
	if (at == ~0U)
	{
		at = m_currentSnap;
	}
	m_snaps.erase(m_snaps.begin() + at);
	m_currentSnap = m_currentSnap % uint32_t(m_snaps.size());
}



}; // namespace chag


