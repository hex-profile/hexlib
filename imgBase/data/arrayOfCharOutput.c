#include "formatting/formatStream.h"
#include "data/array.h"
#include "charType/charType.h"

//================================================================
//
// formatOutput
//
//================================================================

#define TMP_MACRO(Type) \
    \
    template <> \
    void formatOutput(const Type& value, FormatOutputStream& outputStream) \
    { \
        ARRAY_EXPOSE(value); \
        outputStream.write(valuePtr, valueSize); \
    }

TMP_MACRO(Array<const CharType>)
TMP_MACRO(Array<CharType>)
TMP_MACRO(ArrayEx<const CharType*>)
TMP_MACRO(ArrayEx<CharType*>)

#undef TMP_MACRO
