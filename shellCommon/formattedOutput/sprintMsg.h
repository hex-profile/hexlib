#pragma once

#include "formatting/paramMsgOutput.h"
#include "charType/charArray.h"

#include "stlString/stlString.h"
#include "formattedOutput/formatStreamStdio.h"
#include "errorLog/convertAllExceptions.h"

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

template <typename... Types>
inline StlString sprintMsg(const CharArray& format, const Types&... values)
{
    const FormatOutputAtom params[] = {values...};
    ParamMsg paramMsg(format, params, sizeof...(values));
   
    ////

    constexpr size_t bufferSize = 1024;
    CharType bufferArray[bufferSize];
    FormatStreamStdioThunk formatter{bufferArray, bufferSize};
    
    ////

    formatOutput(paramMsg, formatter);
   
    return StlString{formatter.data()};
}
