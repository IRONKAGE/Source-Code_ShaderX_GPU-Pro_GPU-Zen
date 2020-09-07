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
#include "IndirectionTexture.h"

IndirectionTexture::IndirectionTexture (int size)
: stride_ (size)
{
	table_.resize (size * size);
}

void IndirectionTexture::Set (const int page, const std::pair<float, float> uv, const float scale)
{
	table_ [page].u = uv.first;
	table_ [page].v = uv.second;
	table_ [page].scale = scale;
}

const void*	IndirectionTexture::GetData () const
{
	return &table_ [0];
}

int IndirectionTexture::GetDataSize () const
{
	return static_cast<int>(table_.size ()) * sizeof (Entry_t);
}

int IndirectionTexture::GetDataStride () const
{
	return stride_ * sizeof (Entry_t);
}