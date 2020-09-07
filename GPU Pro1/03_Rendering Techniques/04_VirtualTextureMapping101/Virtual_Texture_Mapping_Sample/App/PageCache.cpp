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
#include "PageCache.h"

#include "RenderSystem.h"

#include <D3D10.h>

#include <cassert>

#include "checks.h"

#undef VTM_DEBUG_PAGE_CACHE
#include <iostream>
#include <iomanip>

/////////////////////////////////////////////////////////////////////////////
PageCache::PageCache(ID3D10Device *r, int mipMapCount, int pageSizeRoot, int padding, int cacheSizeRoot)
: device_ (r), mipMapCount_ (mipMapCount)
{
	D3D10_TEXTURE2D_DESC desc;
	::ZeroMemory (&desc, sizeof(desc));

	desc.Width =	cacheSizeRoot;
	desc.Height =	cacheSizeRoot;
	desc.ArraySize = 1;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	desc.Format = DXGI_FORMAT_BC1_UNORM_SRGB; // ; DXGI_FORMAT_R8G8B8A8_UNORM
	desc.Usage = D3D10_USAGE_DEFAULT;
	desc.MipLevels = 1;

	r->CreateTexture2D (&desc, NULL, &texture_);

	pageCountRoot_ = cacheSizeRoot / (pageSizeRoot + padding * 2);
	pageCount_ = pageCountRoot_ * pageCountRoot_;
	realPageSize_ = pageSizeRoot + 2 * padding;
	usablePageSize_ = pageSizeRoot;
	padding_ = padding;
	cacheSize_ = cacheSizeRoot;
	relativePadding_ = static_cast<float>(padding) / cacheSize_;

	slots_.resize (pageCount_);

	Clear ();
}

/////////////////////////////////////////////////////////////////////////////
PageCache::~PageCache ()
{
	texture_->Release ();
}

/////////////////////////////////////////////////////////////////////////////
ID3D10Texture2D*	PageCache::GetTexture () const
{
	return texture_;
}

/////////////////////////////////////////////////////////////////////////////
void PageCache::FreeSlot ()
{
	// Find one slot and free it
	// We only get called if no slot is free
	// So we can search for a good one right away

	CacheHandle_t page = -1;
	int minMipLevel = 16;

	for (int i = 0; i < pageCount_; ++i)
	{
		if (slots_ [i].forced)
		{
			continue;
		}

		if (slots_ [i].mipLevel < minMipLevel)
		{
			minMipLevel = slots_ [i].mipLevel;
			page = i;
		}
	}

	assert (page != -1);
	assert (slots_ [page].forced == false);

	// We must have found something
	freeSlots_.insert (page);
}

/////////////////////////////////////////////////////////////////////////////
bool PageCache::HasFreeSlot () const
{
	return !freeSlots_.empty ();
}

/////////////////////////////////////////////////////////////////////////////
PageCache::CacheHandle_t PageCache::GetNextFreeSlot ()
{
	if (! HasFreeSlot ())
	{
		FreeSlot ();
	}

	// Kinda hacky, but helps, different order for taking out free items
	CacheHandle_t handle = *--freeSlots_.end ();
	freeSlots_.erase (--freeSlots_.end ());

	return handle;
}

/////////////////////////////////////////////////////////////////////////////
PageCache::CacheHandle_t	PageCache::CachePage (
	const D3D10_SUBRESOURCE_DATA* data,
	const int count,
	const PageID_t id,
	const bool forced)
{
	VTM_CHECK(count <= mipMapCount_ && count > 0);

	// Try to restore
	{
		CacheHandle_t tmp;

		if (RestorePage (id, tmp))
		{
			return cachedPages_ [id];
		}
	}

	// Get the next free page
	const CacheHandle_t page = GetNextFreeSlot ();

	assert (freeSlots_.find (page) == freeSlots_.end ());

	cachedPages_ [id] = page;

	if (slots_ [page].valid)
	{
		OnPageDropped (slots_ [page].pageID);

		// Remove it now, otherwise, we leak handles
		cachedPages_.erase (slots_ [page].pageID);
	}

	// Update the slot
	slots_ [page].forced	= forced;
	slots_ [page].mipLevel	= PageID_GetMipMapLevel (id);
	slots_ [page].pageID	= id;
	slots_ [page].valid		= true;

	// Compute the x,y coordinate
	const int x = (page % pageCountRoot_) * realPageSize_;
	const int y = (page / pageCountRoot_) * realPageSize_;

	// Now, do a copy
	for (int i = 0; i < 1; ++i)
	{
		int fx = x / (1 << i);
		int fy = y / (1 << i);

		if (fx > 0) 
		{
			fx = ((fx - 1) / 4 + 1) * 4;
		}

		if (fy > 0)
		{
			fy = ((fy - 1) / 4 + 1) * 4;
		}

		D3D10_BOX box;
		// 0 & 1, otherwise, it is empty ...
		box.front = 0;
		box.back = 1;
		box.top = fy;
		box.bottom = fy + realPageSize_ / (1 << i);
		box.left = fx;
		box.right = fx + realPageSize_ / (1 << i);

		device_->UpdateSubresource (texture_, D3D10CalcSubresource(i, 0, 0), &box,
			data [i].pSysMem, data [i].SysMemPitch, data [i].SysMemSlicePitch);
	}

	return page;
}

/////////////////////////////////////////////////////////////////////////////
std::pair<float, float> PageCache::GetPageCoordinates(const CacheHandle_t handle) const
{
	std::pair<float, float> topLeftCorner (
		((handle % pageCountRoot_) * realPageSize_) / static_cast<float>(cacheSize_),
		((handle / pageCountRoot_) * realPageSize_) / static_cast<float>(cacheSize_)
		);

	// Add the offset
	topLeftCorner.first += static_cast<float>(relativePadding_);
	topLeftCorner.second += static_cast<float>(relativePadding_);

	return topLeftCorner;
}

/////////////////////////////////////////////////////////////////////////////
std::pair<float, float>	PageCache::GetPageSizeInTextureSpace () const
{
	return std::make_pair (
		static_cast<float>(usablePageSize_) / cacheSize_,		
		static_cast<float>(usablePageSize_) / cacheSize_);
}

/////////////////////////////////////////////////////////////////////////////
void PageCache::ReleasePage (const PageID_t id)
{
	// Move it to the free list if possible
	const std::tr1::unordered_map<PageID_t, CacheHandle_t>::const_iterator it =
		cachedPages_.find (id);

	if (it != cachedPages_.end ())
	{
		freeSlots_.insert (it->second);
	}
}

/////////////////////////////////////////////////////////////////////////////
int PageCache::GetPageMipLevel(const PageCache::CacheHandle_t handle) const
{
	return slots_ [handle].mipLevel;
}

/////////////////////////////////////////////////////////////////////////////
void PageCache::Clear ()
{
	cachedPages_.clear ();
	freeSlots_.clear ();

	for (CacheHandle_t i = 0; i < pageCount_; ++i)
	{
		slots_ [i].valid = false;
		freeSlots_.insert (i);
	}
}

/////////////////////////////////////////////////////////////////////////////
void PageCache::SetPageDroppedCallback(std::tr1::function<void(int page,int mipLevel)> callback)
{
	pageDroppedCallback_ = callback;
}

/////////////////////////////////////////////////////////////////////////////
void PageCache::OnPageDropped (const PageID_t p) const
{
	if (pageDroppedCallback_)
	{
		pageDroppedCallback_ (
			PageID_GetPageNumber (p),
			PageID_GetMipMapLevel (p));
	}
}

/////////////////////////////////////////////////////////////////////////////
PageStatus::Enum PageCache::GetPageStatus (const PageID_t id) const
{
	const std::tr1::unordered_map<PageID_t, CacheHandle_t>::const_iterator it =
		cachedPages_.find (id);

	if (it == cachedPages_.end ())
	{
		return PageStatus::Not_Available;
	}
	else
	{
		if (slots_ [it->second].valid == false)
		{
			return PageStatus::Not_Available;
		}

		if (freeSlots_.find (it->second) != freeSlots_.end ())
		{
			return PageStatus::Pending_Delete;
		}
		else
		{
			return PageStatus::Available;
		}
	}
}
	
/////////////////////////////////////////////////////////////////////////////
bool PageCache::RestorePage (const PageID_t id,
							 CacheHandle_t& r)
{
	// Look it up
	const std::tr1::unordered_map<PageID_t, CacheHandle_t>::const_iterator it =
		cachedPages_.find (id);

	if (it == cachedPages_.end ())
	{
		return false;
	}
	else
	{
		VTM_ASSERT (slots_ [it->second].pageID == id);

		// Remove it from the free list
		freeSlots_.erase (it->second);

		r = it->second;

		return true;
	}
}

/////////////////////////////////////////////////////////////////////////////
PageCache::SetContainerType	PageCache::GetPages () const
{
	SetContainerType r;

	for (std::tr1::unordered_map<PageID_t, CacheHandle_t>::const_iterator it = cachedPages_.begin (),
		end = cachedPages_.end ();
		it != end;
		++it)
	{
		if (slots_ [it->second].valid)
		{
			r.insert (it->first);
		}
	}

	return r;
}

/////////////////////////////////////////////////////////////////////////////
void PageCache::GetStatus (int& slotsUsed, int& slotsMarkedFree, int& slotsEmpty)
{
	slotsUsed = slotsMarkedFree = slotsEmpty = 0;

	for (std::vector<Entry_t>::const_iterator it = slots_.begin (), end = slots_.end ();
		it != end; ++it)
	{
		if (it->valid)
		{
			++ slotsUsed;
		}
		else
		{
			++ slotsMarkedFree;
		}
	}

	slotsEmpty = static_cast<int> (freeSlots_.size ());
}