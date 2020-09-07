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
#ifndef VTM_PAGE_CACHE_H
#define VTM_PAGE_CACHE_H

#include <vector>
#include <boost/tr1/unordered_map.hpp>
#include <boost/tr1/unordered_set.hpp>
#include <boost/tr1/functional.hpp>

#include "PageID.h"

struct ID3D10Texture2D;
struct ID3D10Device;
struct D3D10_SUBRESOURCE_DATA;
class RenderSystem;

struct PageStatus
{
	enum Enum
	{
		Available		= 1,	///< Available in the cache
		Not_Available	= 2,	///< Not available
		Pending_Delete	= 4		///< Deleted, but still floating around
	};
};

/**
* The page cache. Used to cache single pages. Uses a LRU scheme to purge the
* last non-used page if possible, and then continues with the lowest-priority
* pages.
*
* Page numbers must use at most 24 bits.
*/
class PageCache
{
public:
	typedef int CacheHandle_t;

public:
	/**
	* The pages are assumend to be squared size, with <c>pageSizeRoot * pageSizeRoot</c> texels.
	*
	* Example: 128, 4, 2048:
	*	Creates a cache with 2048*2048 texels, storing (128+8)² pages
	*/
	PageCache (ID3D10Device* device, int mipMapCount, int pageSizeRoot, int padding, int cacheSizeRoot);
	~ PageCache ();

	/**
	* This function will be called each time a page has been dropped
	*/
	void				SetPageDroppedCallback (std::tr1::function<void (int page, int mipLevel)> callback);

	ID3D10Texture2D*	GetTexture () const;

	/**
	* Cache a page.
	*
	* @pre pageNumber is a 24bit integer
	* @pre count == 2
	*/
	CacheHandle_t	CachePage (
		const D3D10_SUBRESOURCE_DATA* data,
		const int mipMapCount,
		const PageID_t id,
		const bool forced = false);

	/**
	* Remove all entries.
	*/
	void				Clear ();

	/**
	* Get the status for a single page.
	*/
	PageStatus::Enum	GetPageStatus (const PageID_t pageID) const;

	/**
	* Restore a page with the status "Pending delete".
	*
	* @param result The resulting cache handle will be written back here. If the
	*		page could not be restored, the value will not be changed.
	* @return True if the page was restored, else otherwise.
	*/
	bool				RestorePage (const PageID_t id, CacheHandle_t& result);

	/**
	* Get the u,v coordinates for a page.
	*
	* @return first is u, second is v
	*/
	std::pair<float, float>
						GetPageCoordinates (const CacheHandle_t handle) const;

	/**
	* Get the mip-map level for a page.
	*/
	int					GetPageMipLevel (const CacheHandle_t handle) const;

	/**
	* Mark a page as not used.
	*/
	void				ReleasePage (const PageID_t pageID);

	/**
	* Get the size of a page in texture space.
	*
	* @return first is u, second is v
	*/
	std::pair<float, float>
						GetPageSizeInTextureSpace () const;

	typedef std::tr1::unordered_set<PageID_t> SetContainerType;

	/**
	* Get all pages which are currently cached.
	*/
	SetContainerType	GetPages () const;

	/**
	* Get the status of the page cache.
	*
	* @param slotsUsed Slots currently used
	* @param slotsMarkedFree Slots on the free list
	* @param slotsEmpty Slots which have not been touched at all
	*/
	void				GetStatus (int& slotsUsed, int& slotsMarkedFree, int& slotsEmpty);

private:
	// Free one slot. Tries to find an unused texture, then the smallest
	// one (i.e. lowest mipLevel)
	void				FreeSlot ();

	bool				HasFreeSlot () const;

	/**
	* Will purge a page if needed!
	*/
	CacheHandle_t	GetNextFreeSlot ();

	struct Entry_t
	{
		unsigned int	valid		: 1;
		unsigned int	priority	: 2;	///< Reserved
		unsigned int	mipLevel	: 4;	///< 0..15, allows pages up to 32k
		unsigned int	forced		: 1;	///< Never purge this page, useful for top-level pages
		unsigned int	reserved	: 24;	///< Reserved
		int				pageID;
	};

	void					OnPageDropped (const PageID_t pageID) const;


	ID3D10Texture2D*		texture_;
	ID3D10Device*			device_;

	std::vector<Entry_t>	slots_;

	std::tr1::unordered_set
		<
			CacheHandle_t
		>					freeSlots_;

	std::tr1::unordered_map
		<
			PageID_t,
			CacheHandle_t
		>					cachedPages_;

	std::tr1::function<void (int page, int mipLevel)>
							pageDroppedCallback_;

	int						realPageSize_;
	int						usablePageSize_;
	int						cacheSize_;
	int						pageCountRoot_;
	int						padding_;
	int						pageCount_;
	int						mipMapCount_;
	float					relativePadding_;
};

#endif