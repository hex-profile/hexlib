#pragma once

#include "formatting/formatStream.h"

//================================================================
//
// PrettyNumber
//
//================================================================

template <typename Type>
struct PrettyNumber
{
    FormatNumber<Type> number;
};

//================================================================
//
// prettyNumber
//
//================================================================

template <typename Type>
sysinline auto prettyNumber(const FormatNumber<Type>& number)
{
    return PrettyNumber<Type>{number};
}

//================================================================
//
// prettyNumber
//
//================================================================

sysinline auto prettyNumberFp32(float32 value, int precision)
    {return prettyNumber(formatNumber(value, FormatNumberOptions().fformG().precision(precision)));}

////

template <typename Type>
sysinline auto prettyNumber(const Type& value, int precision=3)
    MISSING_FUNCTION_BODY

////

#define TMP_MACRO(Type, _) \
    \
    template <> \
    sysinline auto prettyNumber(const Type& value, int precision) \
        {return prettyNumberFp32(float32(value), precision);}

BUILTIN_INT_FOREACH(TMP_MACRO, _)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, _)

#undef TMP_MACRO
