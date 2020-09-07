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
#include "TP_ParseImage.h"

#include "ParseImage.h"
#include "SparseUsageTable.h"

/////////////////////////////////////////////////////////////////////////////
WorkItem_ParseImage::WorkItem_ParseImage (const byte* image,
										  const int width, const int height,
										  SparseUsageTable* target)
										  : image (image), width (width), height (height),
										  table (target), workItem_ (0)
{

}

/////////////////////////////////////////////////////////////////////////////
WorkItem_ParseImage::~WorkItem_ParseImage ()
{
	if (workItem_)
	{
		::CloseThreadpoolWork (workItem_);
	}
}

/////////////////////////////////////////////////////////////////////////////
void				WorkItem_ParseImage::RunAsync ()
{
	workItem_ = ::CreateThreadpoolWork (&Async, this, NULL);
	::SubmitThreadpoolWork (workItem_);
}

/////////////////////////////////////////////////////////////////////////////
void				WorkItem_ParseImage::Wait ()
{
	::WaitForThreadpoolWorkCallbacks (workItem_, FALSE);
}

/////////////////////////////////////////////////////////////////////////////
VOID CALLBACK WorkItem_ParseImage::Async (
	PTP_CALLBACK_INSTANCE /* Instance */,
	PVOID Context,
	PTP_WORK /* Work */
	)
{
	WorkItem_ParseImage* wid = reinterpret_cast<WorkItem_ParseImage*> (Context);

	ParseImage (wid->image, wid->height, wid->width, wid->table);
}
