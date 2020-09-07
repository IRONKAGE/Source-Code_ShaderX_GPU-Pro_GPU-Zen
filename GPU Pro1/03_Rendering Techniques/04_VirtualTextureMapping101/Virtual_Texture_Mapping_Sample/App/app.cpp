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

/*
 The function to look at is App::Render close to the bottom of this file. Some
 of the calls are asynchronous, and run using the Vista thread-pool, these are
 marked with WorkItem_

 Moreover, some things are delayed by one frame.

 The application initialisation is done in App::Init(), what this does is to
 set up the page-cache, load the heightmap, create the readback buffer and
 load the top-most page into the page cache.
*/

#define _WIN32_WINNT 0x0600 
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "DXUT.h"
#include "app.h"

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <iomanip>
#include <list>
#include <cstdlib>

#include "TileIdRenderAction.h"
#include "TileIDMaterial.h"
#include "TextureInfo.h"
#include "RenderTarget.h"
#include "RenderEntity.h"
#include "RenderWorld.h"
#include "Heightmap.h"
#include "SparseUsageTable.h"
#include "MipMappedTable.h"
#include "IndirectionTexture.h"
#include "TP_ParseImage.h"
#include "TP_TextureFunctions.h"

#undef VTM_DEBUG_APP
#undef VTM_DEBUG_PAGE_CACHE_LOAD

#undef min
using boost::wformat;
using boost::format;

namespace
{
	byte* image = 0;

	const int width = 256;
	const int height = 192;

	MipMapTable<PageCache::CacheHandle_t> pageTable (64);
	std::list<std::tr1::shared_ptr<WorkItem_LoadTexture> > loadRequests;
	
	// Needed for all async work items
	PTP_POOL threadPool;
	
	void PageReleased (int page, int mipLevel)
	{
		struct PageReleasedPredicate
		{
		public:
			PageReleasedPredicate (const PageCache::CacheHandle_t h)
				: handle_ (h)
			{
			}

			bool operator () (const PageCache::CacheHandle_t t) const
			{
				return t == handle_;
			}

		private:
			PageCache::CacheHandle_t handle_;
		};

		const PageCache::CacheHandle_t t = pageTable.Get (page, mipLevel);
		pageTable.Set (page, mipLevel, -1);

		pageTable.SetChildren (page, mipLevel, -1, PageReleasedPredicate (t));
	}

	struct WorkItem_ReleasePages
	{
		WorkItem_ReleasePages (PageCache* cache, SparseUsageTable* table)
			: cache (cache), table (table)
		{
		}

		~WorkItem_ReleasePages ()
		{
			if (workItem_)
			{
				::CloseThreadpoolWork (workItem_);
			}
		}

		void				RunAsync ()
		{
			workItem_ = ::CreateThreadpoolWork (&Async, this, NULL);
			::SubmitThreadpoolWork (workItem_);
		}

		void				Wait ()
		{
			::WaitForThreadpoolWorkCallbacks (workItem_, FALSE);
		}

	private:
		PageCache*			cache;
		SparseUsageTable*	table;

		PTP_WORK			workItem_;
		
		static VOID CALLBACK Async (
			PTP_CALLBACK_INSTANCE Instance,
			PVOID Context,
			PTP_WORK Work
		)
		{
			WorkItem_ReleasePages* wid = reinterpret_cast<WorkItem_ReleasePages*> (Context);

			// Check which pages are cached but not visible
			const PageCache::SetContainerType cachedPages = wid->cache->GetPages ();

			for (PageCache::SetContainerType::const_iterator it = cachedPages.begin (), end = cachedPages.end ();
				it != end;
				++it)
			{
				if (!wid->table->IsUsed (*it) && PageID_GetMipMapLevel (*it) < 6)
				{
					wid->cache->ReleasePage (*it);
				}
			}
		}
	};
}

/////////////////////////////////////////////////////////////////////////////
App::App (ID3D10Device* device)
: device_ (device), renderSystem_ (device), pageCache_ (device, MIP_MAP_COUNT, 128, 4, 2048),
maxPagesToBindPerFrame_ (5), maxItemsToRequestPerFrame_ (20), waitForAllPages_ (false),
useProgressiveLoading_ (true), showCacheStatus_ (false)
{
	pageCache_.SetPageDroppedCallback (PageReleased);
}

/////////////////////////////////////////////////////////////////////////////
void App::SetViewProjectionMatrix (D3DXMATRIX m)
{
	renderSystem_.SetViewProjectionMatrix (m);
}

/////////////////////////////////////////////////////////////////////////////
void App::ShowCacheStatus (bool e)
{
	showCacheStatus_ = e;
}

/////////////////////////////////////////////////////////////////////////////
void App::SetMaximumTileRequestsPerFrame (int limit)
{
	assert (limit >= 0);
	std::cout << "Request limit: " << limit << std::endl;
	maxItemsToRequestPerFrame_ = limit;
}

/////////////////////////////////////////////////////////////////////////////
void App::SetMaximumTilesBoundPerFrame (int limit)
{
	assert (limit >= 0);
	std::cout << "Bind limit: " << limit << std::endl;
	maxPagesToBindPerFrame_ = limit;
}

/////////////////////////////////////////////////////////////////////////////
void App::SetProgressiveLoading (bool enabled)
{
	std::cout << "Precache: " << enabled << std::endl;
	useProgressiveLoading_ = enabled;
}

/////////////////////////////////////////////////////////////////////////////
void App::OnBeginResize ()
{
	renderSystem_.OnBeginResize ();
}

/////////////////////////////////////////////////////////////////////////////
void App::OnFinishResize ()
{
	renderSystem_.OnFinishResize ();
}

/////////////////////////////////////////////////////////////////////////////
void App::Init ()
{
	threadPool = ::CreateThreadpool (NULL);
	::SetThreadpoolThreadMaximum (threadPool, 1);

	delete [] image;
	image = new boost::uint8_t[width * height * 4];
	std::fill (image, image + width * height * 4, 0);

	renderSystem_.Init ();
	renderWorld_ = renderSystem_.CreateRenderWorld ();

	renderTarget_ = renderSystem_.CreateNewRenderTarget (width, height);

	// Create color readback texture
	{
		D3D10_TEXTURE2D_DESC colorReadbackDesc;
		::ZeroMemory (&colorReadbackDesc, sizeof(colorReadbackDesc));

		colorReadbackDesc.Width = width;
		colorReadbackDesc.Height = height;
		colorReadbackDesc.ArraySize = 1;
		colorReadbackDesc.SampleDesc.Count = 1;
		colorReadbackDesc.SampleDesc.Quality = 0;
		colorReadbackDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		colorReadbackDesc.CPUAccessFlags = D3D10_CPU_ACCESS_READ;
		colorReadbackDesc.Usage = D3D10_USAGE_STAGING;
		colorReadbackDesc.MipLevels = 1;

		device_->CreateTexture2D (&colorReadbackDesc, NULL, &colorReadback_);
	}

	// Create indirection texture -- the page table
	{
		D3D10_TEXTURE2D_DESC desc;
		::ZeroMemory (&desc, sizeof(desc));

		desc.Width = 64;
		desc.Height = 64;
		desc.ArraySize = 1;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
		desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
		desc.Usage = D3D10_USAGE_DEFAULT;
		desc.MipLevels = 1;

		device_->CreateTexture2D (&desc, NULL, &indirectionTexture_);
	}

	VirtualTextureSettings_t vts;
	vts.maximumMipMapLevel = 6;
	vts.mipMapScaleFactor = 4;
	vts.pageSize [0] = 128;
	vts.pageSize [1] = 128;

	tileIDMaterial_ = new TileIDMaterial (device_, vts);

	ResetCache ();

	heightmap_ = std::tr1::shared_ptr<Heightmap> (new Heightmap (
		&renderSystem_,
		TEXT("Textures\\height.bmp"),
		pageCache_.GetTexture (),
		indirectionTexture_,
		pageCache_.GetPageSizeInTextureSpace ().first));

	renderWorld_->AddRenderEntity (heightmap_->GetRenderEntity ());
}

/////////////////////////////////////////////////////////////////////////////
void App::ResetCache ()
{
	std::tr1::shared_ptr<WorkItem_LoadTexture> tlr (new WorkItem_LoadTexture (
		"..\\Source-Data\\Tiles\\JPEG\\tile-6-0.jpg",
		PageID_Create (0, 6), MIP_MAP_COUNT));

	tlr->RunAsync ();
	tlr->Wait ();

	// Force loading the top mip-map level
	pageCache_.Clear ();
	pageTable.Clear (-1);
	pageTable.Set (0, 6,
		pageCache_.CachePage (tlr->GetData (), tlr->GetMipMapCount (), tlr->GetId (), true));
}

/////////////////////////////////////////////////////////////////////////////
void App::Shutdown ()
{
	::CloseThreadpool (threadPool);

	delete [] image;
	image = 0;

	colorReadback_->Release ();

	renderSystem_.DestroyRenderWorld (renderWorld_);
	renderSystem_.Shutdown ();

	delete tileIDMaterial_;

	indirectionTexture_->Release ();
}

namespace
{
	/////////////////////////////////////////////////////////////////////////////
	void UpdatePageTable (const PageCache& cache, MipMapTable<PageCache::CacheHandle_t>& pageTable)
	{
		// Update the indirection texture
		for (int i = pageTable.GetMaximumLevel (); i >= 1; --i)
		{
			for (int y = 0; y < pageTable.GetLevelHeight (i); ++y)
			{
				for (int x = 0; x < pageTable.GetLevelWidth (i); ++x)
				{
					// Update the two corresponding elements to this one
					const int lowerX = 2 * x;
					const int lowerY = 2 * y;

					const PageCache::CacheHandle_t thisHandle = pageTable.Get (x, y, i);

					const int thisMipMapLevel = cache.GetPageMipLevel (thisHandle);

					assert (thisHandle != -1);

					if ((pageTable. Get (lowerX, lowerY, i - 1) == -1)
						|| cache.GetPageMipLevel (pageTable. Get (lowerX, lowerY, i - 1)) > thisMipMapLevel)
					{
						pageTable.Set(lowerX, lowerY, i - 1, thisHandle);
					}

					if ((pageTable. Get (lowerX + 1, lowerY, i - 1) == -1)
						|| (cache.GetPageMipLevel (pageTable. Get (lowerX + 1, lowerY, i - 1)) > thisMipMapLevel))
					{
						pageTable.Set(lowerX + 1, lowerY, i - 1, thisHandle);
					}

					if ((pageTable. Get (lowerX, lowerY + 1, i - 1) == -1)
						|| (cache.GetPageMipLevel (pageTable. Get (lowerX, lowerY + 1, i - 1)) > thisMipMapLevel))
					{
						pageTable.Set(lowerX, lowerY + 1, i - 1, thisHandle);
					}

					if ((pageTable. Get (lowerX + 1, lowerY + 1, i - 1) == -1)
						|| (cache.GetPageMipLevel (pageTable. Get (lowerX + 1, lowerY + 1, i - 1)) > thisMipMapLevel))
					{
						pageTable.Set(lowerX + 1, lowerY + 1, i - 1, thisHandle);
					}
				}
			}
		}
	}

	/////////////////////////////////////////////////////////////////////////////
	void UpdateIndirectionTexture (
		ID3D10Device* device,
		ID3D10Texture2D* tex,
		const PageCache& cache,
		const MipMapTable<PageCache::CacheHandle_t>& pageTable
	)
	{
		for (int j = 0; j < 1; ++j)
		{
			// Height too, actually ...
			IndirectionTexture indirectionTexture (pageTable.GetLevelWidth (j)); 
			
			for (int i = 0; i < pageTable.GetLevelElementCount (j); ++i)
			{
				const PageCache::CacheHandle_t handle = pageTable.Get (i, j);

				indirectionTexture.Set (i, cache.GetPageCoordinates (handle), 
					pageTable.GetMaximumLevel () - cache.GetPageMipLevel (handle));
			}

			device->UpdateSubresource (tex,
				D3D10CalcSubresource (j, 0, 0),
				NULL,
				indirectionTexture.GetData (),
				indirectionTexture.GetDataStride (),
				indirectionTexture.GetDataSize ());
		}
	}
}
	
/////////////////////////////////////////////////////////////////////////////
void App::Render (D3DXMATRIX viewProjection)
{
	static SparseUsageTable oldUsageTable (64);
	SparseUsageTable usageTable (64);

	WorkItem_ParseImage parseImage (image, width, height, &usageTable);
	parseImage.RunAsync ();

	WorkItem_ReleasePages releasePages (&pageCache_, &oldUsageTable);
	releasePages.RunAsync ();

	// Perform the tile ID render target readback from the _previous_ frame,
	// see below where the new data is written
	D3D10_MAPPED_TEXTURE2D mappedTexture;
	DXUT_BeginPerfEvent(D3DCOLOR_RGBA(0, 0, 255, 255), L"Render target readback");
	colorReadback_->Map (D3D10CalcSubresource (0, 0, 0), D3D10_MAP_READ, 0, &mappedTexture);

	::memcpy (image, mappedTexture.pData, width * height * 4);

	colorReadback_->Unmap (D3D10CalcSubresource (0, 0, 0));
	DXUT_EndPerfEvent();

	// Render the scene with the page ID shader
	TileIDRenderAction tileIDRenderPass (renderTarget_.get (), tileIDMaterial_);

	DXUT_BeginPerfEvent(D3DCOLOR_RGBA(255, 0, 255, 255), L"Tile ID Render");
	renderWorld_->Render (viewProjection, &tileIDRenderPass);
	DXUT_EndPerfEvent();

	// Copy the page ID render target into the readback texture,
	// will be read in the next frame (see above!)
	DXUT_BeginPerfEvent(D3DCOLOR_RGBA(0, 255, 255, 255), L"Copy render target");
	device_->CopyResource (colorReadback_, renderTarget_->colorRenderTarget);
	DXUT_EndPerfEvent();

	// Do the normal rendering
	DXUT_BeginPerfEvent(D3DCOLOR_RGBA(0, 255, 255, 255), L"Beauty render");
	renderWorld_->Render (viewProjection);
	DXUT_EndPerfEvent();

	// Synchronise
	parseImage.Wait ();
	releasePages.Wait ();
	
	oldUsageTable = usageTable;

	// Find out the items which are not yet cached
	for (SparseUsageTable::ConstIterator it = usageTable.Begin (), end = usageTable.End (); 
		it != end;
		++it)
	{
		// Break if too many pages are requested. Only break if not waiting for all pages
		if (! waitForAllPages_ && (loadRequests.size () > maxItemsToRequestPerFrame_))
		{
			break;
		}

		const PageID_t id = *it;
		const int pageNumber = PageID_GetPageNumber (id);
		const int mipMapLevel = PageID_GetMipMapLevel (id);

		if (pageNumber >= pageTable.GetLevelElementCount (mipMapLevel))
		{
			// Float point error? More serious problem? Sometimes values which are
			// out of bounds appear (y+1, -1)
			// std::cout << "?\n";
			continue;
		}

		if (pageCache_.GetPageStatus (id) == PageStatus::Available)
		{
			continue;
		}

		// Restore page in cache
		if (pageCache_.GetPageStatus (id) == PageStatus::Pending_Delete)
		{
			PageCache::CacheHandle_t h;
			if (pageCache_.RestorePage (id, h))
			{
				pageTable.Set (pageNumber, mipMapLevel, h);
				continue;
			}
		}

		const PageID_t parentId = id;

		const int maxParentMipMapLevel = useProgressiveLoading_ ? pageTable.GetMaximumLevel () : (mipMapLevel + 1);

		// Request the page and all parents
		for (int i = mipMapLevel; i < maxParentMipMapLevel; ++i)
		{
			int x = pageNumber % pageTable.GetLevelWidth (mipMapLevel);
			int y = pageNumber / pageTable.GetLevelHeight (mipMapLevel);

			x >>= (i-mipMapLevel);
			y >>= (i-mipMapLevel);

			const int newNumber = y * pageTable.GetLevelWidth (i) + x; 

			const PageID_t id = PageID_Create (newNumber, i);

			PageCache::CacheHandle_t tmp;
			if (pageCache_.GetPageStatus (id) == PageStatus::Available || pageCache_.RestorePage (id, tmp))
			{
				continue;
			}

			// Check if we already requested this page
			if (pendingPages_.find (id) != pendingPages_.end ())
			{
				continue;
			}

			std::tr1::shared_ptr<WorkItem_LoadTexture> tlr (
				new WorkItem_LoadTexture (
					str (format ("..\\Source-Data\\Tiles\\JPEG\\tile-%1%-%2%.jpg") % i % newNumber),
					id, MIP_MAP_COUNT, id != parentId ? parentId : PageID_CreateInvalid ()));

			tlr->RunAsync ();

			loadRequests.push_back (tlr);
			pendingPages_.insert (id);

			if (waitForAllPages_)
			{
				tlr->Wait ();
			}
		}
	}
	DXUT_EndPerfEvent();

	int boundPages = 0;

	DXUT_BeginPerfEvent(D3DCOLOR_RGBA(0, 255, 255, 255), L"Update page cache");
	std::vector<std::list<std::tr1::shared_ptr<WorkItem_LoadTexture> >::iterator > r;
	for (std::list<std::tr1::shared_ptr<WorkItem_LoadTexture> >::iterator it = loadRequests.begin (), end = loadRequests.end ();
		it != end;
		++it)
	{
		const std::tr1::shared_ptr<WorkItem_LoadTexture> item = *it;

		if (item->IsLoaded () && (waitForAllPages_ || (boundPages < maxPagesToBindPerFrame_)))
		{
			// Check if our parent is a lowest-level request
			const bool ignorePage = 
				// has a parent
				item->HasParent ()
				// and the reason why we are loaded is already loaded
				&& (pageCache_.GetPageStatus (item->GetParentId ()) == PageStatus::Available);

			if (!ignorePage)
			{
				const PageCache::CacheHandle_t handle = pageCache_.CachePage (
					item->GetData (), item->GetMipMapCount (), item->GetId ()
					);

				pageTable.Set (
					PageID_GetPageNumber (item->GetId ()),
					PageID_GetMipMapLevel (item->GetId ()),
					handle);

				++boundPages;
			}		

			// Mark for removal
			r.push_back (it);
		}
	}

	for (std::vector<std::list<std::tr1::shared_ptr<WorkItem_LoadTexture> >::iterator >::iterator it = r.begin (), end = r.end ();
		it != end;
		++it)
	{
		pendingPages_.erase ((**it)->GetId ());
		loadRequests.erase (*it);
	}
	DXUT_EndPerfEvent();

	if (showCacheStatus_)
	{
		int used, free, markedFree;
		pageCache_.GetStatus (used, markedFree, free);

		std::cout << oldUsageTable.GetEntryCount () << "\t" << used << "\t" << free << "\t" << markedFree << "\n";
	}

	UpdatePageTable (pageCache_, pageTable);
	UpdateIndirectionTexture (device_, indirectionTexture_, pageCache_, pageTable);
}