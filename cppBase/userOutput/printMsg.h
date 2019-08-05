#pragma once

#include "formatting/formatModifiers.h"
#include "formatting/paramMsgOutput.h"
#include "prepTools/prepEnum.h"
#include "userOutput/msgLog.h"
#include "compileTools/compileTools.h"

//================================================================
//
// printMsg
//
// Printf-like function for formatted message output.
//
// Usage:
//
// printMsg(msgLog, format, arg0, ..., argN, msgKind)
//
//================================================================

inline bool printMsg(MsgLog& msgLog, const CharArray& format, MsgKind msgKind = msgInfo)
{
    return msgLog.addMsg(format, msgKind);
}

//----------------------------------------------------------------

template <typename Type>
inline void getMsgKind(const Type& value, bool& valid, MsgKind& result)
    {}

template <>
inline void getMsgKind(const MsgKind& value, bool& valid, MsgKind& result)
    {valid = true; result = value;}

//----------------------------------------------------------------

template <typename... Types>
inline bool printMsg(MsgLog& msgLog, const CharArray& format, const Types&... values)
{
    constexpr size_t n = sizeof...(values);
    const FormatOutputAtom params[] = {values...};

    MsgKind msgKind = msgInfo;
    bool msgKindValid = false;
    char tmp[] = {(getMsgKind(values, msgKindValid, msgKind), 0)...};

    ParamMsg paramMsg(format, params, msgKindValid ? n-1 : n);
    return msgLog.addMsg(paramMsg, msgKindValid ? MsgKind(msgKind) : msgInfo);
}
