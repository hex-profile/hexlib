#pragma once

#include "numbers/float/floatBase.h"
#include "formatting/formatNumberOptions.h"
#include "charType/charArray.h"
#include "formatting/formatStream.h"

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
};

//----------------------------------------------------------------

template <typename Type>
sysinline FormatNumberArray<Type> formatNumberArray(const Type* arrayPtr, size_t arraySize, const FormatNumberOptions& options = {}, const CharArray& delimiter = STR(", "))
{
    return {arrayPtr, arraySize, options, delimiter};
}

//================================================================
//
// FormatSimpleArray
//
//================================================================

template <typename Type>
struct FormatSimpleArray
{
    const Type* arrayPtr;
    size_t arraySize;
    CharArray delimiter;
};

//================================================================
//
// FormatSimpleArray
//
//================================================================

template <typename Type>
sysinline FormatSimpleArray<Type> formatSimpleArray(const Type* arrayPtr, size_t arraySize, const CharArray& delimiter = STR(", "))
{
    return {arrayPtr, arraySize, delimiter};
}
