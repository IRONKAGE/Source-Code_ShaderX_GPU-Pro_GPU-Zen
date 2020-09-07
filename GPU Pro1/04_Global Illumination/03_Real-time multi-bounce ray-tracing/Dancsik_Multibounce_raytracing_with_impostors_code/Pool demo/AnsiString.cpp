#include "dxstdafx.h"
#include "AnsiString.h"

AnsiString::AnsiString(const char* a)
{
	length = strlen(a);
	s = new char[length+1];
	StringCchCopyA(s, length+1, a);
}

AnsiString::AnsiString(const wchar_t* w)
{
	length = WideCharToMultiByte(CP_ACP, 0, w, -1, NULL, 0, false, false)-1;
	s = new char[length+1];
	WideCharToMultiByte(CP_ACP, 0, w, -1, s, length+1, false, false);
}

AnsiString::~AnsiString(void)
{
	delete s;
}

AnsiString::AnsiString(const AnsiString& o)
{
	length = o.length;
	s = new char[length+1];
	StringCchCopyA(s, length+1, o.s);
}
/*
const AnsiString& AnsiString::operator=(const AnsiString& o)
{
	if(this != &o)
	{
		delete s;
		length = o.length;
		s = new char[length+1];
		StringCchCopyA(s, length+1, o.s);
	}
	return *this;
}*/

bool AnsiString::operator<(const AnsiString& o) const
{
	return strcmp(s, o.s) < 0;
}

AnsiString::operator const char* () const
{
	return s;
}