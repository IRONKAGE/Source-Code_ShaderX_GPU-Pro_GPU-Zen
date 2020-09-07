#include "dxstdafx.h"
#include "UnicodeString.h"

UnicodeString::UnicodeString(const wchar_t* a)
{
	length = wcslen(a);
	s = new wchar_t[length];
	StringCchCopyW(s, length, a);
}

UnicodeString::UnicodeString(const char* m)
{
	length = MultiByteToWideChar(CP_ACP, 0, m, -1, NULL, 0)-1;
	s = new wchar_t[length+1];
	MultiByteToWideChar(CP_ACP, 0, m, -1, s, length+1);
}

UnicodeString::~UnicodeString(void)
{
	delete s;
}

UnicodeString::UnicodeString(const UnicodeString& o)
{
	length = o.length;
	s = new wchar_t[length+1];
	StringCchCopyW(s, length+1, o.s);
}

/*
const UnicodeString& UnicodeString::operator=(const UnicodeString& o)
{
	if(this != &o)
	{
		delete s;
		length = o.length;
		s = new wchar_t[length+1];
		StringCchCopyW(s, length+1, o.s);
	}
	return *this;
}
*/
bool UnicodeString::operator<(const UnicodeString& o) const
{
	return wcscmp(s, o.s) < 0;
}

UnicodeString::operator const wchar_t*() const
{
	return s;
}