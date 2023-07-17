#include "charArray.h"

#include <string.h>
#include <wchar.h>

//================================================================
//
// charArrayFromPtr
//
//================================================================

template <>
CharArrayEx<char> charArrayFromPtr(const char* cstring)
    {return CharArrayEx<char>(cstring, strlen(cstring));}

template <>
CharArrayEx<wchar_t> charArrayFromPtr(const wchar_t* cstring)
    {return CharArrayEx<wchar_t>(cstring, wcslen(cstring));}
