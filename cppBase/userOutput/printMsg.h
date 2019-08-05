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
inline int8 getMsgKind(const Type& value)
    {return -1;}

template <>
inline int8 getMsgKind(const MsgKind& value)
    {return value;}

//----------------------------------------------------------------

template <typename... Types>
inline bool printMsg(MsgLog& msgLog, const CharArray& format, const Types&... values)
{
    constexpr size_t n = sizeof...(values);
    const FormatOutputAtom params[] = {values...};

    int8 msgKindArray[] = {getMsgKind(values)...};
    int msgKind = msgKindArray[n-1];
    bool msgKindValid = (msgKind != -1);
   
    ParamMsg paramMsg(format, params, msgKindValid ? n-1 : n);
    return msgLog.addMsg(paramMsg, msgKindValid ? MsgKind(msgKind) : msgInfo);
}
