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
#ifndef VTM_VIRTUAL_TEXTURE_H
#define VTM_VIRTUAL_TEXTURE_H

#include "PageCache.h"

class VirtualTexture
{
public:
	VirtualTexture (const int size);
private:
	struct PageTableEntry_t
	{
		PageCache::CacheHandle_t cacheEntry;
	};

	struct IndirectionTableEntry_t
	{
		float	u, v;
		int		mipLevel;
	};

	std::vector<PageTableEntry_t>			pageTable_;
	std::vector<IndirectionTableEntry_t>	indirectionTable_;
};

#endif