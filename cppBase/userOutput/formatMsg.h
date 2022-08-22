#pragma once

#include "userOutput/paramMsg.h"

//================================================================
//
// formatMsg
//
//================================================================

template <typename... Types>
sysinline void formatMsg(FormatOutputStream& stream, CharType specialChar, const CharArray& format, const Types&... values)
{
    const FormatOutputAtom params[] = {values...};
    ParamMsg paramMsg(specialChar, format, params, sizeof...(values));

    FormatOutputAtom& atom = paramMsg;
    stream << atom;
}

//================================================================
//
// formatMsg
//
//================================================================

template <typename... Types>
sysinline void formatMsg(FormatOutputStream& stream, const CharArray& format, const Types&... values)
{
    const FormatOutputAtom params[] = {values...};
    ParamMsg paramMsg(defaultSpecialChar, format, params, sizeof...(values));

    FormatOutputAtom& atom = paramMsg;
    stream << atom;
}
