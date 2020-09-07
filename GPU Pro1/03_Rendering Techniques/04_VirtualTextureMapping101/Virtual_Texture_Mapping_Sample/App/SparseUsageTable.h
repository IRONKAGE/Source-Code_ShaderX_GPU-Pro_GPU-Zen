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
#ifndef VTM_SPARSE_USAGE_TABLE_H
#define VTM_SPARSE_USAGE_TABLE_H

#include <vector>
#include <boost/tr1/unordered_set.hpp>

#include "PageID.h"

struct SparseUsageTableChange_t
{
	enum Change
	{
		Paged_In,
		Paged_Out
	};

	Change	change;
	int		pageNumber;
	int		mipMapLevel;
};

/**
* Maximum mip map level is 16.
*/
class SparseUsageTable
{
	typedef std::tr1::unordered_set<PageID_t>	ContainerType;

public:
	typedef std::tr1::unordered_set<PageID_t> SetContainerType;
	typedef ContainerType::const_iterator ConstIterator;

	/**
	* Table size will be size * size. The size is also used for the maximum
	* mip-map level, which is log2(size).
	*/
	SparseUsageTable (int size);
	~SparseUsageTable ();

	/**
	* Mark a page as used with the given mipMapLevel.
	*/
	void	Set (int pageNumber, int mipMapLevel);

	/**
	* Mark a page as used with the given mipMapLevel.
	*/
	void	Set (int pageX, int pageY, int mipMapLevel);

	/**
	* Clear the table.
	*
	* Marks each page as "not used".
	*/
	void	Clear ();

	bool	IsUsed (const int pageNumber, const int mipMapLevel) const;
	bool	IsUsed (const PageID_t id) const;

	int		GetWidth () const;
	int		GetHeight () const;

	/**
	* @return <c>GetWidth() * GetHeight()</c>
	*/
	int		GetPageCount () const;

	/**
	* Get the number of used entries.
	*/
	int		GetEntryCount () const;

	std::vector<SparseUsageTableChange_t>
			GetChanges (const SparseUsageTable& old) const;

	/**
	* Get all pages which are set in this sparse usage table.
	*/
	SetContainerType	GetPages () const;

	ConstIterator		Begin () const;
	ConstIterator		End () const;

private:
	
	PageID_t					lastID_;
	int							width_,
								height_;
	int							maxMipMapLevel_;
	ContainerType				table_;
};

#endif