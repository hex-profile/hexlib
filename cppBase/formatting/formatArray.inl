#include "formatArray.h"

#include "formatting/formatStream.h"

//================================================================
//
// formatOutputArray
//
//================================================================

template <typename Type>
void formatOutputArray(const FormatArray<Type>& value, FormatOutputStream& outputStream)
{
    for (size_t i = 0; i < value.arraySize; ++i)
    {
        formatOutput(value.arrayPtr[i], outputStream);

        if (i != value.arraySize-1) 
            outputStream.write(STR(", "));
    }
}

//----------------------------------------------------------------

#define FORMAT_OUTPUT_ARRAY_INST(Type) \
    \
    template <> \
    void formatOutput(const FormatArray<Type>& value, FormatOutputStream& outputStream) \
        {formatOutputArray(value, outputStream);}
