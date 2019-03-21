#include "formatStream.h"

#include "formatting/formatModifiers.h"

//================================================================
//
// formatOutput<CharArray>
//
//================================================================

template <>
void formatOutput(const CharArray& value, FormatOutputStream& outputStream)
{
    outputStream.write(value.ptr, value.size);
}

//================================================================
//
// formatOutput<CharType*>
//
//================================================================

template <>
void formatOutput(const CharType* const& value, FormatOutputStream& outputStream)
{
    outputStream.write(charArrayFromPtr(value));
}

template <>
void formatOutput(CharType* const& value, FormatOutputStream& outputStream)
{
    outputStream.write(charArrayFromPtr(value));
}

//================================================================
//
// formatOutput<BuiltinInt>
// formatOutput<BuiltinFloat>
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    template <> \
    void formatOutput(const Type& value, FormatOutputStream& outputStream) \
        {outputStream.write(value);} \
    \
    template <> \
    void formatOutput(const FormatNumber<Type>& value, FormatOutputStream& outputStream) \
        {outputStream.write(value);}

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

//================================================================
//
// formatOutput<void*>
//
//================================================================

template <>
void formatOutput(void* const& value, FormatOutputStream& outputStream)
{
    COMPILE_ASSERT(sizeof(size_t) == sizeof(void*));
    outputStream.write(hex(size_t(value)));
}
