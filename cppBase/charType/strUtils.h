#pragma once

#include <string.h>
#include <wchar.h>

#include "compileTools/compileTools.h"

//================================================================
//
// strLen
//
//================================================================

sysinline size_t strLen(const char* str)
    {return strlen(str);}

sysinline size_t strLen(const wchar_t* str)
    {return wcslen(str);}

//================================================================
//
// strEqual
//
//================================================================

sysinline bool strEqual(const char* a, const char* b)
    {return strcmp(a, b) == 0;}

sysinline bool strEqual(const wchar_t* a, const wchar_t* b)
    {return wcscmp(a, b) == 0;}
