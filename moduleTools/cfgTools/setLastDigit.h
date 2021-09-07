#pragma once

#include "charType/charArray.h"

//================================================================
//
// setLastDigit
//
//================================================================

template <typename Type>
inline CharArrayEx<Type> setLastDigit(Type* ptr, int i)
{
    auto str = charArrayFromPtr(ptr);

    if (str.size >= 1)
        ptr[str.size - 1] = '0' + i;

    return str;
}
