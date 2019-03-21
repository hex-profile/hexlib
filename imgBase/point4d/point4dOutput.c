#include "point4d/point4d.h"

#include "formatting/formatStream.h"

//================================================================
//
// outputPoint
//
//================================================================

template <typename Type>
inline void outputPoint(const Point4D<Type>& value, FormatOutputStream& outputStream)
{
    outputStream.write(value.X);
    outputStream.write(STR(", "));
    outputStream.write(value.Y);
    outputStream.write(STR(", "));
    outputStream.write(value.Z);
    outputStream.write(STR(", "));
    outputStream.write(value.W);
}

//================================================================
//
// outputPoint
//
//================================================================

template <typename Type>
inline void outputPoint(const FormatNumber< Point4D<Type> >& number, FormatOutputStream& outputStream)
{
    outputStream.write(formatNumber(number.value.X, number.options));
    outputStream.write(STR(", "));
    outputStream.write(formatNumber(number.value.Y, number.options));
    outputStream.write(STR(", "));
    outputStream.write(formatNumber(number.value.Z, number.options));
    outputStream.write(STR(", "));
    outputStream.write(formatNumber(number.value.W, number.options));
}

//================================================================
//
// formatOutput< Point4D<Type> >
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    template <> \
    void formatOutput(const Point4D<Type>& value, FormatOutputStream& outputStream) \
        {outputPoint(value, outputStream);} \
    \
    template <> \
    void formatOutput(const FormatNumber< Point4D<Type> >& value, FormatOutputStream& outputStream) \
        {outputPoint(value, outputStream);}

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
