#pragma once

#include "extLib/types/compileTools.h"
#include "extLib/types/charType.h"

//================================================================
//
// CharArrayEx
//
//================================================================

template <typename Type>
class CharArrayEx
{

public:

    HEXLIB_INLINE CharArrayEx(int=0)
        {}

    HEXLIB_INLINE CharArrayEx(const Type* ptr, size_t size)
        : ptr(ptr), size(size) {}

public:

    const Type* ptr = nullptr;
    size_t size = 0;

};

//----------------------------------------------------------------

template <typename Type>
HEXLIB_INLINE auto charArray(const Type* ptr, size_t size)
    {return CharArrayEx<Type>(ptr, size);}

//----------------------------------------------------------------

using CharArray = CharArrayEx<CharType>;
using CharArrayChar = CharArrayEx<char>;
