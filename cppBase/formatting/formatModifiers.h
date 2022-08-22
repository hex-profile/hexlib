#pragma once

#include "formatting/formatStream.h"

//================================================================
//
// dec
// decs
//
// Decimal number of n digits
//
//================================================================

template <typename Type>
sysinline FormatNumber<Type> dec(const Type& value, int32 nDigits)
    {return formatNumber(value, FormatNumberOptions().width(nDigits).alignRight().fillZero());}

template <typename Type>
sysinline FormatNumber<Type> decs(const Type& value, int32 nDigits)
    {return formatNumber(value, FormatNumberOptions().width(nDigits).alignRight().fillZero().plusOn());}

//================================================================
//
// hex
//
// Hexadecimal number of n digits
//
//================================================================

template <typename Type>
sysinline FormatNumber<Type> hex(const Type& value, int32 nDigits = 8)
    {return formatNumber(value, FormatNumberOptions().baseHex().alignRight().fillZero().width(nDigits));}

//================================================================
//
// Float modifiers
//
// fltf / fltg / flte
// The same as ".*f" ".*g" and ".*f" in printf.
//
// fltfs / fltgs / fltes
// The same as previous, but with forced "+" sign for non-negative numbers.
//
//================================================================

#define TMP_MACRO(FF, ff) \
    \
    template <typename Type> \
    sysinline FormatNumber<Type> flt##ff(const Type& value, int32 prec) \
        {return formatNumber(value, FormatNumberOptions().fform##FF().precision(prec));} \
    \
    template <typename Type> \
    sysinline FormatNumber<Type> flt##ff##s(const Type& value, int32 prec) \
        {return formatNumber(value, FormatNumberOptions().fform##FF().precision(prec).plusOn());} \
    \
    template <typename Type> \
    sysinline FormatNumber<Type> flt##ff(const Type& value, char fill, int32 width, int32 prec) \
        {return formatNumber(value, FormatNumberOptions().fform##FF().fillWith(fill).alignInternal().width(width).precision(prec));} \
    \
    template <typename Type> \
    sysinline FormatNumber<Type> flt##ff##s(const Type& value, char fill, int32 width, int32 prec) \
        {return formatNumber(value, FormatNumberOptions().fform##FF().fillWith(fill).alignInternal().width(width).precision(prec).plusOn());}

TMP_MACRO(F, f)
TMP_MACRO(G, g)
TMP_MACRO(E, e)

#undef TMP_MACRO
