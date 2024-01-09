#pragma once

#include "msgLogExKit.h"

#include "errorLog/errorLog.h"
#include "formatting/paramMsgOutput.h"
#include "prepTools/prepEnum.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/msgLog.h"

//================================================================
//
// MsgLogEx
//
// Output a message with trace callstack.
//
//================================================================

struct MsgLogEx
{
    virtual bool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars) =0;
};

//================================================================
//
// MsgLogExNull
//
//================================================================

class MsgLogExNull : public MsgLogEx
{
    virtual bool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars)
        {return true;}
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
    inline bool printMsgTrace \
    ( \
        MsgLogEx& msgLogEx, \
        const CharArray& format, \
        PREP_ENUMERATE_INDEXED_PAIR(n, const T, &v) \
        MsgKind msgKind, \
        stdPars(Kit) \
    ) \
    { \
        const FormatOutputAtom params[COMPILE_CLAMP_MIN(n, 1)] = {PREP_FOR(n, PRINTTRACE__STORE_PARAM, _)}; \
        \
        ParamMsg paramMsg{defaultSpecialChar, format, params, n}; \
        return msgLogEx.addMsgTrace(paramMsg, msgKind, stdPassThru); \
    }

PREP_FOR1(PREP_INC(PRINTTRACE__MAX_COUNT), PRINTTRACE__FUNC, _)

//================================================================
//
// CHECK_TRACE*
//
// If the condition is not true, outputs a formatted message.
// The macro returns allv(condition) value.
//
//================================================================

#define CHECK_TRACE(condition, format) \
    CHECK_EX(condition, printMsgTrace(kit.msgLogEx, format, msgErr, stdPass))

#define CHECK_TRACE1(condition, format, v0) \
    CHECK_EX(condition, printMsgTrace(kit.msgLogEx, format, v0, msgErr, stdPass))

#define CHECK_TRACE2(condition, format, v0, v1) \
    CHECK_EX(condition, printMsgTrace(kit.msgLogEx, format, v0, v1, msgErr, stdPass))

#define CHECK_TRACE3(condition, format, v0, v1, v2) \
    CHECK_EX(condition, printMsgTrace(kit.msgLogEx, format, v0, v1, v2, msgErr, stdPass))

#define CHECK_TRACE4(condition, format, v0, v1, v2, v3) \
    CHECK_EX(condition, printMsgTrace(kit.msgLogEx, format, v0, v1, v2, v3, msgErr, stdPass))

//----------------------------------------------------------------

#define CHECK_TRACE0_THRU(condition, format) \
    CHECK_EX(condition, printMsgTrace(kit.msgLogEx, format, msgErr, stdPassThru))

#define CHECK_TRACE1_THRU(condition, format, v0) \
    CHECK_EX(condition, printMsgTrace(kit.msgLogEx, format, v0, msgErr, stdPassThru))

#define CHECK_TRACE2_THRU(condition, format, v0, v1) \
    CHECK_EX(condition, printMsgTrace(kit.msgLogEx, format, v0, v1, msgErr, stdPassThru))

#define CHECK_TRACE3_THRU(condition, format, v0, v1, v2) \
    CHECK_EX(condition, printMsgTrace(kit.msgLogEx, format, v0, v1, v2, msgErr, stdPassThru))

#define CHECK_TRACE4_THRU(condition, format, v0, v1, v2, v3) \
    CHECK_EX(condition, printMsgTrace(kit.msgLogEx, format, v0, v1, v2, v3, msgErr, stdPassThru))

//================================================================
//
// REQUIRE_TRACE*
//
// Checks a condition; if the condition is not true,
// outputs a formatted message and returns false.
//
//================================================================

#define REQUIRE_TRACE(condition, format) \
    require(CHECK_TRACE(condition, format))

#define REQUIRE_TRACE1(condition, format, v0) \
    require(CHECK_TRACE1(condition, format, v0))

#define REQUIRE_TRACE2(condition, format, v0, v1) \
    require(CHECK_TRACE2(condition, format, v0, v1))

#define REQUIRE_TRACE3(condition, format, v0, v1, v2) \
    require(CHECK_TRACE3(condition, format, v0, v1, v2))

#define REQUIRE_TRACE4(condition, format, v0, v1, v2, v3) \
    require(CHECK_TRACE4(condition, format, v0, v1, v2, v3))

//----------------------------------------------------------------

#define REQUIRE_TRACE_THRU(condition, format) \
    require(CHECK_TRACE0_THRU(condition, format))

#define REQUIRE_TRACE0_THRU(condition, format) \
    require(CHECK_TRACE0_THRU(condition, format))

#define REQUIRE_TRACE1_THRU(condition, format, v0) \
    require(CHECK_TRACE1_THRU(condition, format, v0))

#define REQUIRE_TRACE2_THRU(condition, format, v0, v1) \
    require(CHECK_TRACE2_THRU(condition, format, v0, v1))

#define REQUIRE_TRACE3_THRU(condition, format, v0, v1, v2) \
    require(CHECK_TRACE3_THRU(condition, format, v0, v1, v2))

#define REQUIRE_TRACE4_THRU(condition, format, v0, v1, v2, v3) \
    require(CHECK_TRACE4_THRU(condition, format, v0, v1, v2, v3))
