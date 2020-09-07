#pragma once

class AnsiString
{
	unsigned int length;
	char* s;
public:
	AnsiString(const char* a);
	AnsiString(const wchar_t* a);
	AnsiString(const AnsiString& o);
	~AnsiString(void);

//	const AnsiString& operator=(const AnsiString& o);
	bool operator<(const AnsiString& o) const;

	operator const char* () const;
};
