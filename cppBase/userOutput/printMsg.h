#pragma once

#include "formatting/formatModifiers.h"
#include "formatting/paramMsgOutput.h"
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

sysinline bool printMsg(MsgLog& msgLog, const CharArray& format, MsgKind msgKind = msgInfo)
{
    return msgLog.addMsg(format, msgKind);
}

//================================================================
//
// getMsgKind
//
//================================================================

template <typename Type>
sysinline void getMsgKind(const Type& value, bool& valid, MsgKind& result)
    {}

template <>
sysinline void getMsgKind(const MsgKind& value, bool& valid, MsgKind& result)
    {valid = true; result = value;}

//================================================================
//
// printMsg
//
//================================================================

#define TMP_MACRO(specialChar) \
    { \
        constexpr size_t n = sizeof...(values); \
        const FormatOutputAtom params[] = {values...}; \
        \
        MsgKind msgKind = msgInfo; \
        bool msgKindValid = false; \
        char tmp[] = {(getMsgKind(values, msgKindValid, msgKind), 'x')...}; \
        \
        ParamMsg paramMsg(specialChar, format, params, msgKindValid ? n-1 : n); \
        return msgLog.addMsg(paramMsg, msgKindValid ? MsgKind(msgKind) : msgInfo); \
    }

template <typename... Types>
sysinline bool printMsg(MsgLog& msgLog, const CharArray& format, const Types&... values)
    TMP_MACRO(defaultSpecialChar)

template <typename... Types>
sysinline bool printMsg(MsgLog& msgLog, CharType specialChar, const CharArray& format, const Types&... values)
    TMP_MACRO(specialChar)

#undef TMP_MACRO
