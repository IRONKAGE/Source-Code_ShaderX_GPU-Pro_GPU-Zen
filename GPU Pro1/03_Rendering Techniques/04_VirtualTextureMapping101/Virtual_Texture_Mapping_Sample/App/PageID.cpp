/*
	Virtual texture mapping demo app
    Copyright (C) 2008, 2009 Matthäus G. Chajdas
    Contact: shaderx8@anteru.net

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "PageID.h"

#include "checks.h"

/////////////////////////////////////////////////////////////////////////////
PageID_t PageID_Create (const unsigned int page, int mipLevel)
{
	VTM_CHECK_SLOW ((page & ~0xFFFFFF) == 0 && "Page number must use only 24 bits");
	VTM_CHECK_SLOW ((mipLevel & ~0xFF) == 0 && "Mip-Map level must use only 4 bits");

	// Make sure we only use the lower 24 bit of the page, so the top 4
	// bits are reserved
	return ((page & 0xFFFFFF) << 4) | (mipLevel & 0xF);
}

/////////////////////////////////////////////////////////////////////////////
int	PageID_GetPageNumber (const PageID_t id)
{
	VTM_CHECK_SLOW (PageID_IsValid (id));

	return id >> 4;
}

/////////////////////////////////////////////////////////////////////////////
int PageID_GetMipMapLevel (const PageID_t id)
{
	VTM_CHECK_SLOW (PageID_IsValid (id));

	return id & 0xF;
}

/////////////////////////////////////////////////////////////////////////////
bool PageID_IsValid (const PageID_t page)
{
	return (page >> 31) == 0;	// Top bit must be zero
}

/////////////////////////////////////////////////////////////////////////////
PageID_t PageID_CreateInvalid ()
{
	return 0xFFFFFFFF;
}