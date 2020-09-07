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
#ifndef VTM_MIP_MAPPED_TABLE_H
#define VTM_MIP_MAPPED_TABLE_H

#include <vector>
#include "checks.h"
#include "util.h"

/**
* A class which encapsulates a table at various mip-map levels.
*/
template <typename T>
class MipMapTable
{
public:
	/**
	* Creates a mip-map table, with the level 0 having size*size entries,
	* and level 1 having (size/2)*(size/2) with n levels so that the
	* n-th level has only one entry.
	*
	* @pre Size must be a power of 2
	*/
	MipMapTable (const int size);

	void	Set (const int entry, const int level, const T value);
	void	Set (const int x, const int y, const int level, const T value);
	T		Get (const int entry, const int level) const;
	T		Get (const int x, const int y, const int level) const;


	/**
	* Clear all elements to a user specified clear value.
	*/
	void	Clear (const T clearValue = T ());

	/**
	* Set the children which satisfy a predicate.
	*/
	template <typename Pred>
	void	SetChildren (const int entry, const int level, const T value, Pred predicate);

	template <typename Pred>
	void	SetChildren (const int x, const int y, const int level, const T value, Pred predicate);

	int		GetLevelWidth (const int level) const;
	int		GetLevelHeight (const int level) const;

	/**
	* Get the number of elements on a given level.
	*/
	int		GetLevelElementCount (const int level) const;

	/**
	* Get the maximum level of this table.
	*/
	int		GetMaximumLevel () const;

	/**
	* Get the number of an entry index at a given level.
	*/
	int		GetEntryIndex (const int x, const int y, const int level) const;
private:
	std::vector<T>		table_;
	std::vector<int>	offsets_;
	int					limit_;
};

/////////////////////////////////////////////////////////////////////////////
template <typename T>
MipMapTable<T>::MipMapTable (const int size)
: limit_ (floor_log2 (size))
{
	int accumulator = 0;

	for (int i = 0; i <= limit_; ++i)
	{
		offsets_.push_back (accumulator);
		
		const int size = (1 << (limit_ - i));
		accumulator += size * size;
	}

	table_.resize (accumulator);
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
int MipMapTable<T>::GetEntryIndex (const int x, const int y, const int level) const
{
	return y * GetLevelWidth (level) + x;
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
void MipMapTable<T>::Set(const int entry, const int level, const T value)
{
	VTM_CHECK(entry < GetLevelElementCount (level));
	table_ [offsets_ [level] + entry] = value;
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
T MipMapTable<T>::Get(const int entry, const int level) const
{
	VTM_CHECK(entry < GetLevelElementCount (level));
	return table_ [offsets_ [level] + entry];
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
void MipMapTable<T>::Set(const int x, const int y, const int level, const T value)
{
	Set (GetEntryIndex (x, y, level), level, value);
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
T MipMapTable<T>::Get(const int x, const int y, const int level) const
{
	return Get (GetEntryIndex (x, y, level), level);
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
void MipMapTable<T>::Clear(const T clearValue)
{
	std::fill (table_.begin (), table_.end (), clearValue);
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
int MipMapTable<T>::GetLevelWidth (const int l) const
{
	return (1 << (limit_ - l)); // In one dimension
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
int MipMapTable<T>::GetLevelHeight (const int l) const
{
	return (1 << (limit_ - l)); // In one dimension
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
int MipMapTable<T>::GetLevelElementCount (const int l) const
{
	return GetLevelWidth (l) * GetLevelHeight (l);
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
int MipMapTable<T>::GetMaximumLevel () const
{
	return limit_;
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
template <typename Pred>
void	MipMapTable<T>::SetChildren (const int entry, const int level, const T value, Pred predicate)
{
	if (level == 0)
	{
		return;
	}

	int x = entry % GetLevelWidth (level);
	int y = entry / GetLevelHeight (level);

	int size = 1;
	for (int i = level - 1; i >= 0; --i)
	{
		x *= 2;
		y *= 2;

		size *= 2;

		for (int iy = 0; iy < size; ++iy)
		{
			for (int ix = 0; ix < size; ++ix)
			{
				if (predicate (Get (x + ix, y + iy, i)))
				{
					Set (x + ix, y + iy, i, value);
				}
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////
template <typename T>
template <typename Pred>
void	MipMapTable<T>::SetChildren (const int x, const int y, const int level, const T value, Pred predicate)
{
	SetChildren (GetEntryIndex (x, y, level), level, value, predicate);
}

#endif