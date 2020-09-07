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
#ifndef VTM_TEXTURE_INFO_H
#define VTM_TEXTURE_INFO_H

/**
* Store the information for a single texture.
*/
struct TextureInfo_t
{
	TextureInfo_t (int id, int width, int height);
	
	/**
	* Create an invalid texture info.
	*/
	TextureInfo_t ();

	bool IsValid () const;

	int id;			// Texture ID
	int width;		// Width of this texture
	int height;		// Height of this texture
};

#endif