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
#ifndef VTM_APP_H
#define VTM_APP_H

#include <set>

#include <boost/noncopyable.hpp>

#include "RenderSystem.h"
#include "PageCache.h"

class RenderWorld;
class TileIDMaterial;

class App : public boost::noncopyable
{
public:
	App (ID3D10Device* device);

	void Init ();
	void Shutdown ();

	void SetViewProjectionMatrix (D3DXMATRIX m);

	void OnBeginResize ();

	void OnFinishResize ();

	void ResetCache ();

	void Render (D3DXMATRIX viewProjection);

	void SetMaximumTileRequestsPerFrame (int limit);
	void SetMaximumTilesBoundPerFrame (int limit);
	void SetProgressiveLoading (bool enabled);
	void ShowCacheStatus (bool enabled);

private:
	std::tr1::shared_ptr<class Heightmap>	heightmap_;
	std::tr1::shared_ptr<RenderTarget_t>	renderTarget_;

	RenderSystem					renderSystem_;
	RenderWorld*					renderWorld_;

	ID3D10Device*					device_;

	ID3D10Texture2D*				colorReadback_;
	ID3D10Texture2D*				indirectionTexture_;

	PageCache						pageCache_;

	TileIDMaterial*					tileIDMaterial_;

	int								maxPagesToBindPerFrame_;
	int								maxItemsToRequestPerFrame_;

	bool							waitForAllPages_;
	bool							useProgressiveLoading_;
	bool							showCacheStatus_;

	static const int				MIP_MAP_COUNT = 1;

	std::set<PageID_t>				pendingPages_;
};

#endif