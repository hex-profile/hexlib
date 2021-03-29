#pragma once

#include "userOutput/paramMsg.h"

//================================================================
//
// formatMsg
//
//================================================================

template <typename... Types>
inline void formatMsg(FormatOutputStream& stream, const CharArray& format, const Types&... values)
{
    const FormatOutputAtom params[] = {values...};
    ParamMsg paramMsg(format, params, sizeof...(values));

    FormatOutputAtom& atom = paramMsg;
    formatOutput(atom, stream);
}
