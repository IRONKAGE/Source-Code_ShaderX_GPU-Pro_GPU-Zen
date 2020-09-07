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
#include "ParseImage.h"
#include "SparseUsageTable.h"

void ParseImage (const unsigned char* __restrict image, int height, int width, SparseUsageTable* table)
{
	for (int i = 0; i < height * width; ++i)
	{
		const int offset = i * 4;

		const unsigned char a = image [offset + 3];

		if (a == 0)
		{
			continue;
		}

		const unsigned char r = image [offset];
		const unsigned char g = image [offset + 1];
		const unsigned char b = image [offset + 2];

		table->Set (r, g, b);
	}
}