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
#ifndef VTM_CHECKS_H
#define VTM_CHECKS_H

void VTM_Check_OnFailure (const char* what, const char* file, const int line);

#define VTM_ASSERT(x) do { if (!(x)) VTM_Check_OnFailure (#x, __FILE__, __LINE__); } while (0)

#define VTM_CHECK(x) VTM_ASSERT(x)

#define VTM_CHECK_SLOW(x) 
#endif