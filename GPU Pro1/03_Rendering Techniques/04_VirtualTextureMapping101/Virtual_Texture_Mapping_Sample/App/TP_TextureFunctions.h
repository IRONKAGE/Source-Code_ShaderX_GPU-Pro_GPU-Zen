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
#ifndef VTM_TP_TEXTURE_FUNCTIONS_H
#define VTM_TP_TEXTURE_FUNCTIONS_H

#include <windows.h>
#include <D3D10.h>

#include <string>

#include "PageID.h"

struct WorkItem_LoadTexture
{
public:
	static const int			MAXIMUM_MIP_MAP_LEVELS = 2;

	WorkItem_LoadTexture (const std::string& sourceFile,
		const PageID_t pageId,
		const int mipMapsToLoad,
		const PageID_t parent = PageID_CreateInvalid ());

	~WorkItem_LoadTexture ();

	PageID_t						GetId () const;
	PageID_t						GetParentId () const;
	bool							HasParent () const;
	const D3D10_SUBRESOURCE_DATA*	GetData () const;
	bool							IsLoaded () const;
	int								GetMipMapCount () const;

	// Run using the thread pool
	void							RunAsync ();
	void							Wait ();

private:
	std::string					filename;
	int							mipMaps;
	PageID_t					id;
	D3D10_SUBRESOURCE_DATA		data[MAXIMUM_MIP_MAP_LEVELS];
	void*						data_pointer [MAXIMUM_MIP_MAP_LEVELS];
	PageID_t					parent;
	PTP_WORK					workItem;
	volatile bool				result;	

	static VOID CALLBACK Async (
								PTP_CALLBACK_INSTANCE Instance,
								PVOID Context,
								PTP_WORK Work
								);
};

#endif