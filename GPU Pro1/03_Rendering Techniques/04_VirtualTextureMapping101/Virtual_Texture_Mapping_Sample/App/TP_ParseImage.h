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
#ifndef VTM_TP_PARSE_IMAGE_H
#define VTM_TP_PARSE_IMAGE_H

#include "dxtc.h" // for byte
#include <windows.h>

class SparseUsageTable;

class WorkItem_ParseImage
{
public:
	WorkItem_ParseImage (const byte* image,
		const int width, const int height,
		SparseUsageTable* target);

	~WorkItem_ParseImage ();

	void				RunAsync ();

	void				Wait ();

private:
	const byte*			image;
	int					width;
	int					height;
	SparseUsageTable*	table;

	PTP_WORK			workItem_;

	static VOID CALLBACK Async (
		PTP_CALLBACK_INSTANCE Instance,
		PVOID Context,
		PTP_WORK Work
		);
};
#endif

