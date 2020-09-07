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
#include "SparseUsageTable.h"

#include <cmath>
#include <cassert>

#include "util.h"

//////////////////////////////////////////////////////////////////////////////
SparseUsageTable::SparseUsageTable (int size)
: width_ (size), height_ (size)
{
	maxMipMapLevel_ = floor_log2 (size);
	Clear ();
}

//////////////////////////////////////////////////////////////////////////////
SparseUsageTable::~SparseUsageTable ()
{
}

//////////////////////////////////////////////////////////////////////////////
void SparseUsageTable::Set (int pageNumber, int mipMapLevel)
{
	assert (pageNumber >= 0 && pageNumber < GetPageCount ());

	const PageID_t id = PageID_Create (pageNumber, mipMapLevel);

	if (id == lastID_)
	{
		return;
	}
	else
	{
		lastID_ = id;
	}

	table_.insert (id);
}

//////////////////////////////////////////////////////////////////////////////
void SparseUsageTable::Set (int pageX, int pageY, int mipMapLevel)
{
	const int size = 1 << (maxMipMapLevel_ - mipMapLevel);
	const int coord = pageY * size + pageX;

	Set (coord, mipMapLevel);
}

//////////////////////////////////////////////////////////////////////////////
bool SparseUsageTable::IsUsed (const int pageNumber, const int mipMapLevel) const
{
	return IsUsed (PageID_Create (pageNumber, mipMapLevel));
}

//////////////////////////////////////////////////////////////////////////////
bool SparseUsageTable::IsUsed (const PageID_t id) const
{
	return table_.find (id) != table_.end ();
}

//////////////////////////////////////////////////////////////////////////////
void SparseUsageTable::Clear ()
{
	lastID_ = PageID_CreateInvalid ();
	
	table_.clear ();
}

//////////////////////////////////////////////////////////////////////////////
int SparseUsageTable::GetWidth () const
{
	return width_;
}

//////////////////////////////////////////////////////////////////////////////
int SparseUsageTable::GetHeight () const
{
	return height_;
}

//////////////////////////////////////////////////////////////////////////////
int SparseUsageTable::GetPageCount () const
{
	return GetWidth () * GetHeight ();
}

//////////////////////////////////////////////////////////////////////////////
int SparseUsageTable::GetEntryCount () const
{
	return static_cast<int> (table_.size ());
}

//////////////////////////////////////////////////////////////////////////////
SparseUsageTable::SetContainerType SparseUsageTable::GetPages () const
{
	SetContainerType r (table_.begin (), table_.end ());

	return r;
}

//////////////////////////////////////////////////////////////////////////////
std::vector<SparseUsageTableChange_t>
SparseUsageTable::GetChanges (const SparseUsageTable& old) const
{
	assert (width_ == old.GetWidth ());
	assert (height_ == old.GetHeight ());

	std::vector<SparseUsageTableChange_t> result;

	// Walk over the old ones. Elements not present in the new one are paged
	// out
	for (ConstIterator it =
		old.table_.begin (), end = old.table_.end (); it != end;
		++it)
	{
		if (table_.find (*it) == table_.end ())
		{
			SparseUsageTableChange_t sutc;
			sutc.change = SparseUsageTableChange_t::Paged_Out;
			sutc.pageNumber = PageID_GetPageNumber (*it);
			sutc.mipMapLevel = PageID_GetMipMapLevel (*it);

			result.push_back (sutc);
		}
	}

	// Walk over the new ones. Elements not present in the old one are paged
	// in
	for (ConstIterator it =
		table_.begin (), end = table_.end (); it != end;
		++it)
	{
		if (old.table_.find (*it) == old.table_.end ())
		{
			SparseUsageTableChange_t sutc;
			sutc.change = SparseUsageTableChange_t::Paged_In;
			sutc.pageNumber = PageID_GetPageNumber (*it);
			sutc.mipMapLevel = PageID_GetMipMapLevel (*it);

			result.push_back (sutc);
		}
	}

	return result;
}

//////////////////////////////////////////////////////////////////////////////
SparseUsageTable::ConstIterator SparseUsageTable::Begin () const
{
	return table_.begin ();
}

//////////////////////////////////////////////////////////////////////////////
SparseUsageTable::ConstIterator SparseUsageTable::End () const
{
	return table_.end ();
}