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

/**
* This code is the C-part from http://cache-www.intel.com/cd/00/00/32/43/324337_324337.pdf
*
* Real-Time DXT Compression
* May 20th 2006 J.M.P. van Waveren
*/

#include "dxtc.h"
#include <cstdlib>
#include <cstring>

#define ALIGN16(v) __declspec(align(16)) v

void ExtractBlock( const byte *inPtr, int width, byte *colorBlock )
{
	for ( int j = 0; j < 4; j++ ) 
	{
		::memcpy( &colorBlock[j*4*4], inPtr, 4*4 );
		inPtr += width * 4;
	}
}

word ColorTo565 (const byte *color) 
{
	return ((color [0] >> 3) << 11) | ((color [1] >> 2) << 5) | (color [2] >> 3);
}

void EmitByte(  Buffer_t* buffer, byte b ) 
{
	buffer->pointer [0] = b;
	buffer->pointer += 1;
}

void EmitWord( Buffer_t* buffer, word s ) 
{
	buffer->pointer[0] = ( s >> 0 ) & 255;
	buffer->pointer[1] = ( s >> 8 ) & 255;
	buffer->pointer += 2;
}

void EmitDoubleWord( Buffer_t* buffer, dword i ) 
{
	buffer->pointer[0] = ( i >> 0 ) & 255;
	buffer->pointer[1] = ( i >> 8 ) & 255;
	buffer->pointer[2] = ( i >> 16 ) & 255;
	buffer->pointer[3] = ( i >> 24 ) & 255;
	buffer->pointer += 4;
}

int ColorDistance( const byte *c1, const byte *c2 ) 
{
	return ( ( c1[0] - c2[0] ) * ( c1[0] - c2[0] ) ) +
		( ( c1[1] - c2[1] ) * ( c1[1] - c2[1] ) ) +
		( ( c1[2] - c2[2] ) * ( c1[2] - c2[2] ) );
}

void SwapColors( byte *c1, byte *c2 ) 
{
	byte tm[3];
	memcpy( tm, c1, 3 );
	memcpy( c1, c2, 3 );
	memcpy( c2, tm, 3 );
}

void GetMinMaxColors( const byte *colorBlock, byte *minColor, byte *maxColor ) 
{
	int maxDistance = -1;
	for ( int i = 0; i < 64 - 4; i += 4 ) {
		for ( int j = i + 4; j < 64; j += 4 ) {
			int distance = ColorDistance( &colorBlock[i], &colorBlock[j] );
			if ( distance > maxDistance ) {
				maxDistance = distance;
				::memcpy( minColor, colorBlock+i, 3 );
				::memcpy( maxColor, colorBlock+j, 3 );
			}
		}
	}
	if ( ColorTo565( maxColor ) < ColorTo565( minColor ) ) 
	{
		SwapColors( minColor, maxColor );
	}
}

#define C565_5_MASK 0xF8 // 0xFF minus last three bits
#define C565_6_MASK 0xFC // 0xFF minus last two bits

void EmitColorIndices( Buffer_t* buffer, const byte *colorBlock, const byte *minColor, const byte *maxColor ) 
{
	byte colors[4][4];
	unsigned int indices[16];

	colors[0][0] = ( maxColor[0] & C565_5_MASK ) | ( maxColor[0] >> 5 );
	colors[0][1] = ( maxColor[1] & C565_6_MASK ) | ( maxColor[1] >> 6 );
	colors[0][2] = ( maxColor[2] & C565_5_MASK ) | ( maxColor[2] >> 5 );
	colors[1][0] = ( minColor[0] & C565_5_MASK ) | ( minColor[0] >> 5 );
	colors[1][1] = ( minColor[1] & C565_6_MASK ) | ( minColor[1] >> 6 );
	colors[1][2] = ( minColor[2] & C565_5_MASK ) | ( minColor[2] >> 5 );
	colors[2][0] = ( 2 * colors[0][0] + 1 * colors[1][0] ) / 3;
	colors[2][1] = ( 2 * colors[0][1] + 1 * colors[1][1] ) / 3;
	colors[2][2] = ( 2 * colors[0][2] + 1 * colors[1][2] ) / 3;
	colors[3][0] = ( 1 * colors[0][0] + 2 * colors[1][0] ) / 3;
	colors[3][1] = ( 1 * colors[0][1] + 2 * colors[1][1] ) / 3;
	colors[3][2] = ( 1 * colors[0][2] + 2 * colors[1][2] ) / 3;
	for ( int i = 0; i < 16; i++ ) 
	{
		unsigned int minDistance = INT_MAX;
		
		for ( int j = 0; j < 4; j++ ) 
		{
			unsigned int dist = ColorDistance( &colorBlock[i*4], &colors[j][0] );

			if ( dist < minDistance ) 
			{
				minDistance = dist;
				indices[i] = j;
			}
		}
	}

	dword result = 0;
	
	for ( int i = 0; i < 16; i++ ) 
	{
		result |= ( indices[i] << (unsigned int)( i << 1 ) );
	}

	EmitDoubleWord( buffer, result );
}

void CompressImageDXT1( const byte *inBuf, Buffer_t* outBuf, int width, int height, int* outputBytes ) 
{
	ALIGN16( byte block[64] );
	ALIGN16( byte minColor[4] );
	ALIGN16( byte maxColor[4] );

	for ( int j = 0; j < height; j += 4, inBuf += width * 4*4 ) 
	{
		for ( int i = 0; i < width; i += 4 ) 
		{
			ExtractBlock ( inBuf + i * 4, width, block );
			GetMinMaxColors( block, minColor, maxColor );
			EmitWord ( outBuf, ColorTo565( maxColor ) );
			EmitWord ( outBuf, ColorTo565( minColor ) );
			EmitColorIndices ( outBuf, block, minColor, maxColor );
		}
	}

	*outputBytes = outBuf->pointer - outBuf->buffer;
}