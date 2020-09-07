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
#include "MipMapCreator.h"


std::tr1::shared_ptr<Image> Resize (std::tr1::shared_ptr<Image> source)
{
	const int elementSize = Image::GetBytesPerPixel (source->GetFormat ());
	/*
	* TODO: Run over one scanline, and combine two values
	*		On the next run, combine the next two and merge
	*/
	if (source->GetWidth () > 1 && source->GetHeight () > 1)
	{
		unsigned char* newImage = new unsigned char[source->GetHeight () / 2 * source->GetWidth () / 2 * elementSize];

		const int widthHalf = source->GetWidth () / 2;
		const int heightHalf = source->GetHeight () / 2;

		const unsigned char* sd = static_cast<const unsigned char*> (source->GetData ());

		for (int y = 0; y < heightHalf; ++y)
		{
			for (int x = 0; x < widthHalf; ++x)
			{
				const int d_o = ((y * widthHalf) + x) * elementSize;
				const int s_o_0 = (2 * y * source->GetWidth () + x * 2) * elementSize;
				const int s_o_1 = ((2 * y + 1) * source->GetWidth () + x * 2) * elementSize;
								
				for (int i = 0; i < elementSize; ++i)
				{
					unsigned short v = 0;
					
					v += sd [s_o_0 + i];
					v += sd [s_o_0 + i + elementSize];
					v += sd [s_o_1 + i];
					v += sd [s_o_1 + i + elementSize];

					newImage [d_o + i] = v / 4;
				}
			}
		}

		return std::tr1::shared_ptr<Image> (new Image (newImage, source->GetWidth () / 2, source->GetHeight () / 2, source->GetFormat ()));
	}
	else
	{
		throw std::exception ("Cannot resize 1x1 image");
	}
}