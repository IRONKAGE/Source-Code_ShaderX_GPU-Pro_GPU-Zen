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
#ifndef VTM_PAGE_ID_H
#define VTM_PAGE_ID_H

/**
* PageIDs are used throughout the whole system. A page ID consists of a page
* number, which must be 24 bit large (allowing for 2^12 pages in both x and y),
* and a MipMap level, which must be 4 bit (0..15, which covers all mip-maps from
* 0.. 2^15). The page id should be treated as an opaque data type, there is no
* guarantee about how the bits are ordered or which bits are set.
*/

typedef unsigned int PageID_t;

/**
* Page fits in 24 bit.
* MipLevel fits into 4 bit.
*/
PageID_t		PageID_Create (const unsigned int page, const int mipLevel);
int				PageID_GetPageNumber (const PageID_t id);
int				PageID_GetMipMapLevel (const PageID_t id);
bool			PageID_IsValid (const PageID_t);
PageID_t		PageID_CreateInvalid ();

#endif