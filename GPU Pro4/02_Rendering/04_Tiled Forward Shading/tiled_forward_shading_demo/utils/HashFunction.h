#ifndef __HashFunction_h_
#define __HashFunction_h_

#include "IntTypes.h"
#include <string.h>
#include <stdlib.h>
#include <string> 

typedef uint32_t HashID;

#if 0
inline HashID genericDataHash(const uint8 *ptr, size_t bytes)
{
	const uint32_t magic = 0x9E3779B9;
  HashID result = 0;
  // a potentially worthless hashing mechanism for arbitrary vertex data...
	for (size_t j = 0; j < bytes; ++j)
	{
		if (result & (1 << 31))
		{
			result = (result * 2 + ptr[j]) ^ magic;
		}
		else
		{
			result = result * 2 + ptr[j];
		}
	}
	return result;
}

#else

inline HashID genericDataHash(const uint8 *data, size_t bytes)
{
	// 'm' and 'r' are mixing constants generated offline.
	// They're not really 'magic', they just happen to work well.

	const unsigned int m = 0x5bd1e995;
	const int r = 24;

	uint32_t len = uint32_t(bytes);
	// Initialize the hash to a 'random' value

	HashID h = 0 ^ len;

	// Mix 4 bytes at a time into the hash
	while(len >= 4)
	{
		unsigned int k = *(unsigned int *)data;

		k *= m; 
		k ^= k >> r; 
		k *= m; 
		
		h *= m; 
		h ^= k;

		data += 4;
		len -= 4;
	}
	
	// Handle the last few bytes of the input array

	switch(len)
	{
	case 3: h ^= data[2] << 16;
	case 2: h ^= data[1] << 8;
	case 1: h ^= data[0];
	        h *= m;
	};

	// Do a few final mixes of the hash to ensure the last few
	// bytes are well-incorporated.

	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;

	return h;
} 
#endif 

// used to map      
template <typename KEY_T>
struct HashFunction
{
  HashID hash(const KEY_T &key)
  {
    return HashID(key);
  }


  bool isEqual(const KEY_T &key1, const KEY_T &key2)
  {
    return key1 == key2;
  }
};



template <>
inline HashID HashFunction<const char *>::hash(char const * const &key)
{
  HashID h = 0;
  const char *str = key;
  const char* end = str + strlen(str);
  while (str < end)
  {
#ifdef WIN32
    h = _lrotr(h , 3);
#else // !WIN32
    for(int i = 0; i < 3; ++i)
    {
      bool carry = (h & 1) != 0;
      h >>= 1;
      if (carry)
        h |= 0x80000000UL;
    }
#endif // WIN32
    h += *str++;
  }
  return h;
}



template <>
inline HashID HashFunction<std::string>::hash(const std::string &key)
{
  HashID h = 0;
  const char *str = key.c_str();
  const char* end = str + key.length();
  while (str < end)
  {
#ifdef WIN32
    h = _lrotr(h , 3);
#else // !WIN32
    for(int i = 0; i < 3; ++i)
    {
      bool carry = (h & 1) != 0;
      h >>= 1;
      if (carry)
        h |= 0x80000000UL;
    }
#endif // WIN32
    h += *str++;
  }
  return h;
}



// comparison for c strings
template <>
inline bool HashFunction<const char *>::isEqual(char const * const &key1, char const * const &key2)
{
  return strcmp(key1, key2) == 0;
}


#endif // __HashFunction_h_
