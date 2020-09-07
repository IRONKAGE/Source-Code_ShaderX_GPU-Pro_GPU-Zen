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
#include "TP_TextureFunctions.h"
#include "ImageIO.h"
#include "MipMapCreator.h"
#include "DXTC.h"
#include "checks.h"

//////////////////////////////////////////////////////////////////////////////
WorkItem_LoadTexture::WorkItem_LoadTexture (const std::string& sourceFile,
		const PageID_t pageId,
		const int mipMapsToLoad,
		const PageID_t parent)
		: filename (sourceFile), id (pageId), mipMaps (mipMapsToLoad), parent (parent),
		result (false), workItem (0)
{
	for (int i = 0; i < mipMaps; ++i)
	{
		data_pointer [i] = 0;
	}
}

//////////////////////////////////////////////////////////////////////////////
WorkItem_LoadTexture::~WorkItem_LoadTexture ()
{
	for (int i = 0; i < mipMaps; ++i)
	{
		std::free (data_pointer [i]);
	}

	if (workItem)
	{
		::CloseThreadpoolWork (workItem);
	}
}

//////////////////////////////////////////////////////////////////////////////
int WorkItem_LoadTexture::GetMipMapCount () const
{
	return mipMaps;
}

//////////////////////////////////////////////////////////////////////////////
PageID_t WorkItem_LoadTexture::GetId () const
{
	return id;
}

//////////////////////////////////////////////////////////////////////////////
PageID_t WorkItem_LoadTexture::GetParentId () const
{
	return parent;
}

//////////////////////////////////////////////////////////////////////////////
bool WorkItem_LoadTexture::HasParent () const
{
	return PageID_IsValid (parent);
}

//////////////////////////////////////////////////////////////////////////////
const D3D10_SUBRESOURCE_DATA* WorkItem_LoadTexture::GetData () const
{
	return data;
}

//////////////////////////////////////////////////////////////////////////////
bool WorkItem_LoadTexture::IsLoaded () const
{
	return result;
}

//////////////////////////////////////////////////////////////////////////////
void WorkItem_LoadTexture::RunAsync ()
{
	workItem = ::CreateThreadpoolWork (&Async, this, NULL);
	::SubmitThreadpoolWork (workItem);
}

//////////////////////////////////////////////////////////////////////////////
void WorkItem_LoadTexture::Wait ()
{
	::WaitForThreadpoolWorkCallbacks (workItem, FALSE);
}

//////////////////////////////////////////////////////////////////////////////
VOID CALLBACK WorkItem_LoadTexture::Async (
		PTP_CALLBACK_INSTANCE /* Instance */,
		PVOID Context,
		PTP_WORK /* Work */
		)
	{
		WorkItem_LoadTexture* lt = reinterpret_cast<WorkItem_LoadTexture*> (Context);

		VTM_ASSERT(lt->mipMaps <= WorkItem_LoadTexture::MAXIMUM_MIP_MAP_LEVELS && lt->mipMaps > 0);
		VTM_ASSERT(! lt->result);

		std::tr1::shared_ptr<Image> im = LoadJPEG (lt->filename.c_str ());
		im = ChangeFormat (im, Image::RGBA_32);

		Buffer_t buffer;
		buffer.size = im->GetHeight () * im->GetWidth () / 2;
		buffer.buffer =  reinterpret_cast<byte*> (std::malloc (buffer.size));
		buffer.pointer = buffer.buffer;

		for (int i = 0; i < lt->mipMaps; ++i)
		{
			int size = -1;
			CompressImageDXT1 (static_cast<const byte*> (im->GetData ()), &buffer, im->GetWidth (), im->GetHeight (), &size);
			VTM_ASSERT(size > 0);

			lt->data_pointer [i] = std::malloc (size);
			lt->data [i].pSysMem = lt->data_pointer [i];
			lt->data [i].SysMemPitch = im->GetWidth () * 2;
			lt->data [i].SysMemSlicePitch = size;

			::memcpy (lt->data_pointer [i], buffer.buffer, size);

			im = Resize (im);
			buffer.pointer = buffer.buffer;
		}

		std::free (buffer.buffer);

		lt->result = true;
	}

	