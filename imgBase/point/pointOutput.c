#include "point/point.h"

#include "formatting/formatStream.h"

//================================================================
//
// outputPoint
//
//================================================================

template <typename Type>
inline void outputPoint(const Point<Type>& value, FormatOutputStream& outputStream)
{
    outputStream.write(value.X);
    outputStream.write(STR(", "));
    outputStream.write(value.Y);
}

//================================================================
//
// outputPoint
//
//================================================================

template <typename Type>
inline void outputPoint(const FormatNumber< Point<Type> >& number, FormatOutputStream& outputStream)
{
    outputStream.write(formatNumber(number.value.X, number.options));
    outputStream.write(STR(", "));
    outputStream.write(formatNumber(number.value.Y, number.options));
}

//================================================================
//
// formatOutput< Point<Type> >
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    template <> \
    void formatOutput(const Point<Type>& value, FormatOutputStream& outputStream) \
        {outputPoint(value, outputStream);} \
    \
    template <> \
    void formatOutput(const FormatNumber< Point<Type> >& value, FormatOutputStream& outputStream) \
        {outputPoint(value, outputStream);}

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
