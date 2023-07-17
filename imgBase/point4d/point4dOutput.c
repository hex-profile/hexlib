#include "point4d/point4d.h"

#include "formatting/formatStream.h"

//================================================================
//
// outputPoint
//
//================================================================

template <typename Type>
sysinline void outputPoint(const Point4D<Type>& value, FormatOutputStream& outputStream)
{
    outputStream
        << value.X << STR(", ")
        << value.Y << STR(", ")
        << value.Z << STR(", ")
        << value.W;
}

//================================================================
//
// outputPoint
//
//================================================================

template <typename Type>
sysinline void outputPoint(const FormatNumber<Point4D<Type>>& number, FormatOutputStream& outputStream)
{
    outputStream
        << formatNumber(number.value.X, number) << STR(", ")
        << formatNumber(number.value.Y, number) << STR(", ")
        << formatNumber(number.value.Z, number) << STR(", ")
        << formatNumber(number.value.W, number);
}

//================================================================
//
// formatOutput<Point4D<Type>>
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    template <> \
    void formatOutput(const Point4D<Type>& value, FormatOutputStream& outputStream) \
        {outputPoint(value, outputStream);} \
    \
    template <> \
    void formatOutput(const FormatNumber<Point4D<Type>>& value, FormatOutputStream& outputStream) \
        {outputPoint(value, outputStream);}

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
