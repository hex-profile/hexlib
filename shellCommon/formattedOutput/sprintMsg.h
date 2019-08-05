#pragma once

#include <sstream>

#include "formatting/paramMsgOutput.h"
#include "prepTools/prepEnum.h"
#include "charType/charArray.h"

#include "stlString/stlString.h"
#include "formattedOutput/formatStreamStl.h"

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

inline StlString printMsg(const CharArray& format)
{
    return StlString(format.ptr, format.size);
}

//----------------------------------------------------------------

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
