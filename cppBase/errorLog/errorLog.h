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
    virtual bool isThreadProtected() const =0;
    virtual void addErrorSimple(const CharType* message) =0;
    virtual void addErrorTrace(const CharType* message, TRACE_PARAMS(trace)) =0;
};

//================================================================
//
// ErrorLogNull
//
//================================================================

class ErrorLogNull : public ErrorLog
{

public:

    bool isThreadProtected() const override
        {return true;}

    void addErrorSimple(const CharType* message) override
        {}

    void addErrorTrace(const CharType* message, TRACE_PARAMS(trace)) override
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

#define CHECK_TRACE_PREFIX \
    TRACE_AUTO_LOCATION CT(": ")

#define CHECK_TRACE(condition, messageLiteral) \
    CHECK_EX(condition, kit.errorLog.addErrorTrace(CHECK_TRACE_PREFIX messageLiteral, TRACE_PASSTHRU(stdTraceName)))

//----------------------------------------------------------------

#define CHECK_FAIL_MSG(condition) \
    CT(TRACE_STRINGIZE(condition)) CT(" failed")

//----------------------------------------------------------------

#define CHECK(condition) \
    CHECK_TRACE(condition, CHECK_FAIL_MSG(condition))

//================================================================
//
// REQUIRE
//
// Checks a condition; if the condition is not true,
// outputs an error message to the error log and returns false.
//
//================================================================

#define REQUIRE_EX(condition, failReport) \
    require(CHECK_EX(condition, failReport))

//----------------------------------------------------------------

#define REQUIRE(condition) \
    require(CHECK(condition))
