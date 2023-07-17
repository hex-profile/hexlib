#include "point/point.h"

#include "formatting/formatStream.h"

//================================================================
//
// outputPoint
//
//================================================================

template <typename Type>
sysinline void outputPoint(const Point<Type>& value, FormatOutputStream& outputStream)
{
    outputStream
        << value.X << STR(", ")
        << value.Y;
}

//================================================================
//
// outputPoint
//
//================================================================

template <typename Type>
sysinline void outputPoint(const FormatNumber<Point<Type>>& number, FormatOutputStream& outputStream)
{
    outputStream
        << formatNumber(number.value.X, number) << STR(", ")
        << formatNumber(number.value.Y, number);
}

//================================================================
//
// formatOutput<Point<Type>>
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    template <> \
    void formatOutput(const Point<Type>& value, FormatOutputStream& outputStream) \
        {outputPoint(value, outputStream);} \
    \
    template <> \
    void formatOutput(const FormatNumber<Point<Type>>& value, FormatOutputStream& outputStream) \
        {outputPoint(value, outputStream);}

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
