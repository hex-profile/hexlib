#pragma once

#include <sstream>

#include "formatting/paramMsgOutput.h"
#include "charType/charArray.h"

#include "stlString/stlString.h"
#include "formattedOutput/formatStreamStl.h"
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
   
    std::basic_stringstream<CharType> stringStream;
    FormatStreamStlThunk formatToStream(stringStream);
   
    formatOutput(paramMsg, formatToStream);
   
    return stringStream.rdbuf()->str();
}
