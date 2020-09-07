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
#ifndef VTM_IMAGE_IO_H
#define VTM_IMAGE_IO_H

#include <boost/tr1/memory.hpp>
#include <boost/noncopyable.hpp>

class Image : private boost::noncopyable
{
public:
	enum Format
	{
		RGB_24,
		RGBA_32
	};

	Image (unsigned char* data, int width, int height, Format format);
	~ Image ();

	const void*	GetData () const;
	int			GetWidth () const;
	int			GetHeight () const;
	Format		GetFormat () const;
	int			GetDataSize () const;

	static int	GetBytesPerPixel (Format f);

private:
	unsigned char*	data_;
	int				width_;
	int				height_;
	Format			format_;
};

std::tr1::shared_ptr<Image> ChangeFormat (std::tr1::shared_ptr<Image> in, Image::Format target);
std::tr1::shared_ptr<Image> LoadJPEG (const char* filename);

std::tr1::shared_ptr<Image> PadImage (std::tr1::shared_ptr<Image> in, const int minSize = 4);

#endif