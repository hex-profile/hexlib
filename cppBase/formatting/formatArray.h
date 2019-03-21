#pragma once

#include "formatting/formatStream.h"

//================================================================
//
// FormatArray
//
// An array of numeric values with options.
//
//================================================================

template <typename Type>
struct FormatArray
{
    const Type* arrayPtr;
    size_t arraySize;

    inline FormatArray(const Type* arrayPtr, size_t arraySize)
        : arrayPtr(arrayPtr), arraySize(arraySize) {}
};

//----------------------------------------------------------------

template <typename Type>
inline FormatArray<Type> formatArray(const Type* arrayPtr, size_t arraySize)
{
    return FormatArray<Type>(arrayPtr, arraySize);
}
