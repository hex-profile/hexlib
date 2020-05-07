#include "formatNumberArray.h"

#include "formatting/formatStream.h"

//================================================================
//
// outputNumberArray
//
//================================================================

template <typename Type>
sysinline void outputNumberArray(const Type* arrayPtr, size_t arraySize, const FormatNumberOptions& options, FormatOutputStream& outputStream)
{
    for_count (i, arraySize)
    {
        outputStream.write(FormatNumber<Type>(arrayPtr[i], options));
        if (i != arraySize-1) outputStream.write(STR(", "));
    }
}

//================================================================
//
// formatOutput<FormatNumberArray>
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    template <> \
    void formatOutput(const FormatNumberArray<Type>& value, FormatOutputStream& outputStream) \
        {outputNumberArray(value.arrayPtr, value.arraySize, value.options, outputStream);}

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
