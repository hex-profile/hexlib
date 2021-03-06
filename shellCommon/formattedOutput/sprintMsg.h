#pragma once

#include "formatting/paramMsgOutput.h"
#include "charType/charArray.h"

#include "stlString/stlString.h"
#include "formatting/messageFormatter.h"

//================================================================
//
// sprintMsg
//
// Printf-like function, producing STL string.
//
// Usage:
//
// sprintMsg(format, arg0, ..., argN)
//
//================================================================

template <typename Kit, typename... Types>
inline StlString sprintMsg(const Kit& kit, const CharArray& format, const Types&... values)
{
    const FormatOutputAtom params[] = {values...};
    ParamMsg paramMsg(format, params, sizeof...(values));
   
    ////

    kit.formatter.clear();
    formatOutput(paramMsg, kit.formatter);

    if_not (kit.formatter.valid())
        throw std::bad_alloc();
   
    return StlString{kit.formatter.data()};
}
