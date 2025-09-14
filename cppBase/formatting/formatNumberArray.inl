#include "formatNumberArray.h"

#include "formatting/formatStream.h"

//================================================================
//
// outputNumberArray
//
//================================================================

template <typename Type>
sysinline void outputNumberArray(const Type* arrayPtr, size_t arraySize, const FormatNumberOptions& options, const CharArray& delimiter,  FormatOutputStream& outputStream)
{
    for_count (i, arraySize)
    {
        outputStream << formatNumber(arrayPtr[i], options);
        if (i != arraySize-1) outputStream << delimiter;
    }
}

//================================================================
//
// FORMAT_NUMBER_ARRAY_INSTANTIATE
//
//================================================================

#define FORMAT_NUMBER_ARRAY_INSTANTIATE(Type, _) \
    \
    template <> \
    void formatOutput(const FormatNumberArray<Type>& value, FormatOutputStream& outputStream) \
        {outputNumberArray(value.arrayPtr, value.arraySize, value.options, value.delimiter, outputStream);}

//================================================================
//
// outputSimpleArray
//
//================================================================

template <typename Type>
sysinline void outputSimpleArray(const Type* arrayPtr, size_t arraySize, const CharArray& delimiter,  FormatOutputStream& outputStream)
{
    for_count (i, arraySize)
    {
        outputStream << arrayPtr[i];
        if (i != arraySize-1) outputStream << delimiter;
    }
}

//================================================================
//
// FORMAT_SIMPLE_ARRAY_INSTANTIATE
//
//================================================================

#define FORMAT_SIMPLE_ARRAY_INSTANTIATE(Type, _) \
    \
    template <> \
    void formatOutput(const FormatSimpleArray<Type>& value, FormatOutputStream& outputStream) \
        {outputSimpleArray(value.arrayPtr, value.arraySize, value.delimiter, outputStream);}
