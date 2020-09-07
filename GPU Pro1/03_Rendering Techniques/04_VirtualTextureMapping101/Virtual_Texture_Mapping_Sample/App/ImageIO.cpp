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
#include "ImageIO.h"
#include <iostream>

extern "C"
{
	#include "../Extern-libs/jpeg-6b/jpeglib.h"
}

#include <cstdio>

Image::Image (unsigned char* data, int width, int height, Format format)
: data_ (data), width_ (width), height_ (height), format_ (format)
{
}

Image::~Image ()
{
	delete [] data_;
}

const void*	Image::GetData () const
{
	return data_;
}

int			Image::GetWidth () const
{
	return width_;
}

int			Image::GetHeight () const
{
	return height_;
}

Image::Format	Image::GetFormat () const
{
	return format_;
}

int Image::GetDataSize () const
{
	return GetBytesPerPixel (GetFormat ()) * GetWidth () * GetHeight ();
}

int Image::GetBytesPerPixel (Format f)
{
	switch (f)
	{
	case RGB_24:
		return 3;
	case RGBA_32:
		return 4;
	}

	return -1;
}

std::tr1::shared_ptr<Image> LoadJPEG (const char* filename)
{
	jpeg_decompress_struct cinfo;
	jpeg_error_mgr errorManager;

	cinfo.err = jpeg_std_error (&errorManager);
	jpeg_create_decompress (&cinfo);

	std::FILE* handle = std::fopen (filename, "rb");
	jpeg_stdio_src (&cinfo, handle);

	jpeg_read_header (&cinfo, TRUE);

	jpeg_start_decompress (&cinfo);

	// Memory will be deleted by Image
	unsigned char* image_data = new unsigned char [cinfo.output_width * cinfo.output_height * cinfo.output_components];

	while (cinfo.output_scanline < cinfo.output_height)
	{ 
		int offset = cinfo.output_components * cinfo.output_width * cinfo.output_scanline;

		unsigned char* startPointer = image_data + offset;
		unsigned char** row_array = &startPointer;
		jpeg_read_scanlines(&cinfo, row_array, 1);
	}

	jpeg_finish_decompress (&cinfo);
	std::fclose (handle);

	jpeg_destroy_decompress (&cinfo);

	return std::tr1::shared_ptr<Image> (new Image (image_data, cinfo.output_width, cinfo.output_height, Image::RGB_24));
}

std::tr1::shared_ptr<Image> ChangeFormat (std::tr1::shared_ptr<Image> in, Image::Format targetFormat)
{
	const int size = in->GetWidth () * in->GetHeight () * Image::GetBytesPerPixel (targetFormat);
	unsigned char* target = new unsigned char [size];
	
	const int equalChannels = std::min (Image::GetBytesPerPixel (targetFormat), Image::GetBytesPerPixel (in->GetFormat ()));
	const int differentChannels = std::max (Image::GetBytesPerPixel (targetFormat) - Image::GetBytesPerPixel (in->GetFormat ()), 0);

	for (int y = 0; y < in->GetHeight (); ++y)
	{
		for (int x = 0; x < in->GetWidth (); ++x)
		{
			const int s_o = (y * in->GetWidth () + x) * Image::GetBytesPerPixel (in->GetFormat ());
			const int d_o = (y * in->GetWidth () + x) * Image::GetBytesPerPixel (targetFormat);
		
			for (int i = 0; i < equalChannels; ++i)
			{
				(target + d_o)[i] = (static_cast<const unsigned char*> (in->GetData ()) + s_o)[i];
			}

			for (int i = 0; i < differentChannels; ++i)
			{
				(target + d_o)[i + equalChannels] = 0;
			}
		}
	}

	return std::tr1::shared_ptr<Image> (new Image (target, in->GetWidth (), in->GetHeight (), targetFormat));
}
std::tr1::shared_ptr<Image> PadImage (std::tr1::shared_ptr<Image> in, const int minSize)
{
	int width = in->GetWidth ();
	int height = in->GetHeight ();

	width = ((width - 1) / minSize + 1) * minSize;
	height = ((height - 1) / minSize + 1) * minSize;

	if (width == in->GetWidth () && height == in->GetHeight ())
	{
		return in;
	}

	const int bpp = Image::GetBytesPerPixel (in->GetFormat ());

	const int padH = width - in->GetWidth ();
	const int padV = height - in->GetHeight ();

	unsigned char* newImage = new unsigned char [width * height * bpp];

	for (int y = 0; y < in->GetHeight (); ++y)
	{
		const int s_o = y * in->GetWidth () * bpp;
		const int d_o = y * width * bpp;
		// Copy as much of the line as possible
		::memcpy (newImage + d_o, static_cast<const unsigned char*> (in->GetData ()) + s_o, in->GetWidth () * bpp); 
		// Now copy the last element until we filled up
		for (int i = 0; i < padH; ++i)
		{
			::memcpy (newImage + in->GetWidth () * bpp + i * bpp, newImage + (in->GetWidth () - 1) * bpp, bpp);
		}
	}

	// Now copy the remaining scanlines down
	for (int i = 0; i < padV; ++i)
	{
		const int s_o = (in->GetHeight () - 1) * width * bpp;
		const int d_o = (in->GetHeight () + i) * width * bpp;

		::memcpy (newImage + d_o, newImage + s_o, width * bpp);
	}

	return std::tr1::shared_ptr<Image> (new Image (newImage, width, height, in->GetFormat ()));
}