#pragma once

#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"
#include "errorLog/debugBreak.h"

//================================================================
//
// ErrorLog
//
// Basic error log interface, widely used, ASSERT-like.
//
// Custom virtual func implementation: The function call is compiled
// VERY often, so it should take as little instructions as possible.
//
//================================================================

struct ErrorLog
{

public:

    inline bool isThreadProtected() const
        {return threadProtectedFunc(*this);}

    inline void addErrorSimple(const CharType* message)
        {addErrorSimpleFunc(*this, message);}

    inline void addErrorTrace(const CharType* message, TRACE_PARAMS(trace))
        {addErrorTraceFunc(*this, message, TRACE_PASSTHRU(trace));}

private:

    typedef bool ThreadProtected(const ErrorLog& self);
    ThreadProtected* const threadProtectedFunc;

    typedef void AddErrorSimple(ErrorLog& self, const CharType* message);
    AddErrorSimple* const addErrorSimpleFunc;

    typedef void AddErrorTrace(ErrorLog& self, const CharType* message, TRACE_PARAMS(trace));
    AddErrorTrace* const addErrorTraceFunc;

public:

    inline ErrorLog(ThreadProtected* threadProtectedFunc, AddErrorSimple* addErrorSimpleFunc, AddErrorTrace* addErrorTraceFunc)
        :
        threadProtectedFunc(threadProtectedFunc),
        addErrorSimpleFunc(addErrorSimpleFunc),
        addErrorTraceFunc(addErrorTraceFunc)
    {
    }

};

//================================================================
//
// ErrorLogNull
//
//================================================================

class ErrorLogNull : public ErrorLog
{

public:

    inline ErrorLogNull()
        : ErrorLog(isThreadProtected, addErrorSimple, addErrorTrace) {}

    static bool isThreadProtected(const ErrorLog& self)
        {return true;}

    static void addErrorSimple(ErrorLog& self, const CharType* message)
        {}

    static void addErrorTrace(ErrorLog& self, const CharType* message, TRACE_PARAMS(trace))
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
    (allv(condition) || ((failReport, DEBUG_BREAK_INLINE()), false))

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
// REQUIRE is for bool functions.
// REQUIREV is for void functions.
//
//================================================================

#define REQUIRE_EX(condition, failReport) \
    require(CHECK_EX(condition, failReport))

#define REQUIREV_EX(condition, failReport) \
    requirev(CHECK_EX(condition, failReport))

//----------------------------------------------------------------

#define REQUIRE(condition) \
    require(CHECK(condition))

#define REQUIREV(condition) \
    requirev(CHECK(condition))
