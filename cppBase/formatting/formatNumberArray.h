#pragma once

#include "numbers/float/floatBase.h"
#include "formatting/formatNumberOptions.h"
#include "charType/charArray.h"

//================================================================
//
// FormatNumberArray
//
// An array of numeric values with options.
//
//================================================================

template <typename Type>
struct FormatNumberArray
{
    const Type* arrayPtr;
    size_t arraySize;
    FormatNumberOptions options;
    CharArray delimiter;

    inline FormatNumberArray(const Type* arrayPtr, size_t arraySize, const FormatNumberOptions& options, const CharArray& delimiter)
        : arrayPtr(arrayPtr), arraySize(arraySize), options(options), delimiter(delimiter) {}
};

//----------------------------------------------------------------

template <typename Type>
inline FormatNumberArray<Type> formatNumberArray(const Type* arrayPtr, size_t arraySize, const FormatNumberOptions& options = {}, const CharArray& delimiter = STR(", "))
{
    return FormatNumberArray<Type>(arrayPtr, arraySize, options, delimiter);
}
