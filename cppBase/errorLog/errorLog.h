#pragma once

#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"

//================================================================
//
// ErrorLog
//
// Basic error log interface, widely used, ASSERT-like.
//
//================================================================

struct ErrorLog
{
    virtual void addErrorTrace(const CharType* message, TRACE_PARAMS(trace)) =0;
};

//================================================================
//
// ErrorLogNull
//
//================================================================

class ErrorLogNull : public ErrorLog
{
    virtual void addErrorTrace(const CharType* message, TRACE_PARAMS(trace))
        {}
};

//================================================================
//
// CHECK
//
// If the condition is not true, outputs an error message to error log.
// The macro returns allv(condition) value.
//
// Used very often: ~6 instructions, not including the condition check.
//
//================================================================

#define CHECK_EX(condition, failReport) \
    (allv(condition) || (failReport, false))

//----------------------------------------------------------------

#define CHECK_PREFIX \
    TRACE_AUTO_LOCATION CT(": ")

#define CHECK_CUSTOM(condition, messageLiteral) \
    CHECK_EX(condition, kit.errorLog.addErrorTrace(CHECK_PREFIX messageLiteral, TRACE_PASSTHRU(trace)))

//----------------------------------------------------------------

#define CHECK_FAIL_MSG(condition) \
    CT(TRACE_STRINGIZE(condition)) CT(" failed")

//----------------------------------------------------------------

#define CHECK(condition) \
    CHECK_CUSTOM(condition, CHECK_FAIL_MSG(condition))

//================================================================
//
// REQUIRE
//
//================================================================

#define REQUIRE(condition) \
    require(CHECK(condition))

//================================================================
//
// REQUIRE_CUSTOM
//
//================================================================

#define REQUIRE_CUSTOM(condition, messageLiteral) \
    require(CHECK_CUSTOM(condition, messageLiteral))

//================================================================
//
// REQUIRE_EX
//
//================================================================

#define REQUIRE_EX(condition, failReport) \
    require(CHECK_EX(condition, failReport))
