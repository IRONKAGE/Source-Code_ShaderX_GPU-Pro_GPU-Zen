#pragma once

class UnicodeString
{
	unsigned int length;
	wchar_t* s;
public:
	UnicodeString(const wchar_t* a);
	UnicodeString(const char* m);
	UnicodeString(const UnicodeString& o);
	~UnicodeString(void);

//	const UnicodeString& operator=(const UnicodeString& o);
	bool operator<(const UnicodeString& o) const;
	
	operator const wchar_t*() const;
};
