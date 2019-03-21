#pragma once

#include <string.h>

#include "charType/charArray.h"

//================================================================
//
// strEqual
//
//================================================================

sysinline bool strEqual(const CharArray& a, const CharArray& b)
{
    return
        a.size == b.size &&
        memcmp(a.ptr, b.ptr, a.size * sizeof(CharType)) == 0;
}
