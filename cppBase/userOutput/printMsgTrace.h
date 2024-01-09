#pragma once

#include "errorLog/errorLog.h"
#include "formatting/paramMsgOutput.h"
#include "prepTools/prepEnum.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/msgLog.h"
#include "userOutput/msgLogExKit.h"

//================================================================
//
// MsgLogEx
//
// Output a message with trace callstack.
//
//================================================================

struct MsgLogEx
{
    virtual stdbool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdParsNull) =0;
};

//================================================================
//
// MsgLogExNull
//
//================================================================

class MsgLogExNull : public MsgLogEx
{
    virtual stdbool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdParsNull)
        {returnTrue;}
};

//================================================================
//
// printMsgTrace
//
// The same as printMsg, but the message is outputted with trace callstack.
//
//================================================================

#define PRINTTRACE__MAX_COUNT 4

//----------------------------------------------------------------

#define PRINTTRACE__STORE_PARAM(n, _) \
    v##n,

#define PRINTTRACE__FUNC(n, _) \
    \
    template <PREP_ENUMERATE_INDEXED(n, typename T) typename Kit> \
    sysinline stdbool printMsgTrace \
    ( \
        const CharArray& format, \
        PREP_ENUMERATE_INDEXED_PAIR(n, const T, &v) \
        MsgKind msgKind, \
        stdPars(Kit) \
    ) \
    { \
        const FormatOutputAtom params[COMPILE_CLAMP_MIN(n, 1)] = {PREP_FOR(n, PRINTTRACE__STORE_PARAM, _)}; \
        \
        ParamMsg paramMsg{defaultSpecialChar, format, params, n}; \
        require(kit.msgLogEx.addMsgTrace(paramMsg, msgKind, stdPassThru)); \
        returnTrue; \
    }

PREP_FOR1(PREP_INC(PRINTTRACE__MAX_COUNT), PRINTTRACE__FUNC, _)

//================================================================
//
// REQUIRE_TRACE*
//
// Checks a condition; if the condition is not true,
// outputs a formatted message and returns error.
//
//================================================================

#define REQUIRE_TRACE_HELPER(condition, printBody) \
    do { \
        if (!allv(condition)) \
        { \
            require(printBody); \
            returnFalse; \
        } \
    } while (0)

//----------------------------------------------------------------

#define REQUIRE_TRACE(condition, format) \
    REQUIRE_TRACE_HELPER(condition, printMsgTrace(format, msgErr, stdPass))

#define REQUIRE_TRACE1(condition, format, v0) \
    REQUIRE_TRACE_HELPER(condition, printMsgTrace(format, v0, msgErr, stdPass))

#define REQUIRE_TRACE2(condition, format, v0, v1) \
    REQUIRE_TRACE_HELPER(condition, printMsgTrace(format, v0, v1, msgErr, stdPass))

#define REQUIRE_TRACE3(condition, format, v0, v1, v2) \
    REQUIRE_TRACE_HELPER(condition, printMsgTrace(format, v0, v1, v2, msgErr, stdPass))

#define REQUIRE_TRACE4(condition, format, v0, v1, v2, v3) \
    REQUIRE_TRACE_HELPER(condition, printMsgTrace(format, v0, v1, v2, v3, msgErr, stdPass))

//----------------------------------------------------------------

#define REQUIRE_TRACE_THRU(condition, format) \
    REQUIRE_TRACE_HELPER(condition, printMsgTrace(format, msgErr, stdPassThru))

#define REQUIRE_TRACE1_THRU(condition, format, v0) \
    REQUIRE_TRACE_HELPER(condition, printMsgTrace(format, v0, msgErr, stdPassThru))

#define REQUIRE_TRACE2_THRU(condition, format, v0, v1) \
    REQUIRE_TRACE_HELPER(condition, printMsgTrace(format, v0, v1, msgErr, stdPassThru))

#define REQUIRE_TRACE3_THRU(condition, format, v0, v1, v2) \
    REQUIRE_TRACE_HELPER(condition, printMsgTrace(format, v0, v1, v2, msgErr, stdPassThru))

#define REQUIRE_TRACE4_THRU(condition, format, v0, v1, v2, v3) \
    REQUIRE_TRACE_HELPER(condition, printMsgTrace(format, v0, v1, v2, v3, msgErr, stdPassThru))
