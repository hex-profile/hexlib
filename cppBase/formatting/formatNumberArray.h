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

    inline FormatNumberArray(const Type* arrayPtr, size_t arraySize, const FormatNumberOptions& options)
        : arrayPtr(arrayPtr), arraySize(arraySize), options(options) {}
};

//----------------------------------------------------------------

template <typename Type>
inline FormatNumberArray<Type> formatNumberArray(const Type* arrayPtr, size_t arraySize, const FormatNumberOptions& options)
{
    return FormatNumberArray<Type>(arrayPtr, arraySize, options);
}
