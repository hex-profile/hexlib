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

#define SPRINTMSG__MAX_COUNT 4

//----------------------------------------------------------------

inline StlString printMsg(const CharArray& format)
{
    return StlString(format.ptr, format.size);
}

//----------------------------------------------------------------

#define SPRINTMSG__STORE_PARAM(n, _) \
    v##n,

#define SPRINTMSG__PRINT_MSG(n, _) \
    \
    template <PREP_ENUM_INDEXED(n, typename T)> \
    inline StlString sprintMsg(const CharArray& format, PREP_ENUM_INDEXED_PAIR(n, const T, &v)) \
    { \
        const FormatOutputAtom params[] = {PREP_FOR(n, SPRINTMSG__STORE_PARAM, _)}; \
        ParamMsg paramMsg(format, params, n); \
        \
        std::basic_stringstream<CharType> stringStream; \
        FormatStreamStlThunk formatToStream(stringStream); \
        \
        formatOutput(paramMsg, formatToStream); \
        \
        return stringStream.rdbuf()->str(); \
    }

#define SPRINTMSG__PRINT_MSG_THUNK(n, _) \
    SPRINTMSG__PRINT_MSG(PREP_INC(n), _)

PREP_FOR1(SPRINTMSG__MAX_COUNT, SPRINTMSG__PRINT_MSG_THUNK, _)
