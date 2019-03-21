#pragma once

#include "formatting/formatModifiers.h"
#include "formatting/paramMsgOutput.h"
#include "prepTools/prepEnum.h"
#include "userOutput/msgLog.h"

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

#define PRINTMSG__MAX_COUNT 10

//----------------------------------------------------------------

inline bool printMsg(MsgLog& msgLog, const CharArray& format, MsgKind msgKind = msgInfo)
{
    return msgLog.addMsg(format, msgKind);
}

//----------------------------------------------------------------

#define PRINTMSG__STORE_PARAM(n, _) \
    v##n,

#define PRINTMSG__PRINT_MSG(n, _) \
    \
    template <PREP_ENUM_INDEXED(n, typename T)> \
    inline bool printMsg \
    ( \
        MsgLog& msgLog, \
        const CharArray& format, \
        PREP_ENUM_INDEXED_PAIR(n, const T, &v), \
        MsgKind msgKind = msgInfo \
    ) \
    { \
        const FormatOutputAtom params[] = {PREP_FOR(n, PRINTMSG__STORE_PARAM, _)}; \
        \
        ParamMsg paramMsg(format, params, n); \
        return msgLog.addMsg(paramMsg, msgKind); \
    }

#define PRINTMSG__PRINT_MSG_THUNK(n, _) \
    PRINTMSG__PRINT_MSG(PREP_INC(n), _)

PREP_FOR1(PRINTMSG__MAX_COUNT, PRINTMSG__PRINT_MSG_THUNK, _)
