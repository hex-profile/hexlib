#pragma once

#include "errorLogExKit.h"

#include "prepTools/prepEnum.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/msgLog.h"
#include "formatting/paramMsgOutput.h"
#include "errorLog/errorLog.h"

//================================================================
//
// ErrorLogEx
//
// Output a message with trace callstack.
//
//================================================================

struct ErrorLogEx
{
    virtual bool isThreadProtected() const =0;

    virtual bool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars) =0;
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
        ErrorLogEx& errorLogEx, \
        const CharArray& format, \
        PREP_ENUMERATE_INDEXED_PAIR(n, const T, &v) \
        MsgKind msgKind, \
        stdPars(Kit) \
    ) \
    { \
        const FormatOutputAtom params[COMPILE_CLAMP_MIN(n, 1)] = {PREP_FOR(n, PRINTTRACE__STORE_PARAM, _)}; \
        \
        ParamMsg paramMsg(format, params, n); \
        return errorLogEx.addMsgTrace(paramMsg, msgKind, stdPassThru); \
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

#define CHECK_TRACE0(condition, format) \
    CHECK_EX(condition, printMsgTrace(kit.errorLogEx, format, msgErr, stdPass))

#define CHECK_TRACE1(condition, format, v0) \
    CHECK_EX(condition, printMsgTrace(kit.errorLogEx, format, v0, msgErr, stdPass))

#define CHECK_TRACE2(condition, format, v0, v1) \
    CHECK_EX(condition, printMsgTrace(kit.errorLogEx, format, v0, v1, msgErr, stdPass))

#define CHECK_TRACE3(condition, format, v0, v1, v2) \
    CHECK_EX(condition, printMsgTrace(kit.errorLogEx, format, v0, v1, v2, msgErr, stdPass))

#define CHECK_TRACE4(condition, format, v0, v1, v2, v3) \
    CHECK_EX(condition, printMsgTrace(kit.errorLogEx, format, v0, v1, v2, v3, msgErr, stdPass))

//================================================================
//
// REQUIRE_TRACE*
//
// Checks a condition; if the condition is not true,
// outputs a formatted message and returns false.
//
//================================================================

#define REQUIRE_TRACE(condition, format) \
    require(CHECK_TRACE0(condition, format))

#define REQUIRE_TRACE0(condition, format) \
    require(CHECK_TRACE0(condition, format))

#define REQUIRE_TRACE1(condition, format, v0) \
    require(CHECK_TRACE1(condition, format, v0))

#define REQUIRE_TRACE2(condition, format, v0, v1) \
    require(CHECK_TRACE2(condition, format, v0, v1))

#define REQUIRE_TRACE3(condition, format, v0, v1, v2) \
    require(CHECK_TRACE3(condition, format, v0, v1, v2))

#define REQUIRE_TRACE4(condition, format, v0, v1, v2, v3) \
    require(CHECK_TRACE4(condition, format, v0, v1, v2, v3))
