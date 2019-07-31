#pragma once

#include <cstring>

//================================================================
//
// strEqual
//
//================================================================

sysinline bool strEqual(const char* a, const char* b)
    {return strcmp(a, b) == 0;}

sysinline bool strEqual(const wchar_t* a, const wchar_t* b)
    {return wcscmp(a, b) == 0;}
