#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// Boolean Flag Reference Error Handling Mode
//
//----------------------------------------------------------------
//
// This experimental error handling method through dragging a reference to a
// boolean flag contains controversial decisions, but leads to cleaner and
// shorter application code, similar to that achieved using exceptions.
//
// It does not require enclosing each function call in an error checking
// `require` macro, which was particularly inconvenient for large calls:
//
// require
// (
//     kit.gpuImageConsole.addMatrixEx
//     (
//         rawImage,
//         float32(typeMin<MonoPixel>()),
//         float32(typeMax<MonoPixel>()),
//         point(1.f), INTERP_NONE, rawImageSize, BORDER_ZERO,
//         paramMsg(STR("View %, Raw Input"), args.displayedView),
//         stdPass
//     )
// );
//
//----------------------------------------------------------------
//
// How it works: each function, in addition to the standard set of parameters
// (kit, callstack, profiler), receives an additional reference to a boolean
// flag within `stdPars`.
//
// The flag is initially created with the value `true` (when creating root
// standard parameters) and remains `true` during normal program execution.
//
// The flag is passed between functions by reference. If an error occurs in a
// function, it sets the flag to `false` and returns. The calling function
// checks the flag and, if it is reset, also returns.
//
//----------------------------------------------------------------
//
// In this error handling mode, `stdPass` becomes a "very bad macro (tm)". It:
//
// * Closes the function call bracket;
//
// * Performs a success flag check, and, if the flag is reset, returns from the
//   function;
//
// * Ends with something like `do {...} while (0`, so the user code can close
//   the function call bracket and `stdPass` just looks like the last function
//   parameter.
//
//----------------------------------------------------------------
//
// This approach eliminates boilerplate return checks but has side effects. In
// particular, the `stdPass*` macro is no longer a single entity and splits the
// function call into two statements.
//
// As a result, if a function call is under an `if` or `while` statement, only
// the first part of the macro may be executed, and the check is performed
// after the statement.
//
// At first glance, it seems that this prevents practical use of this method.
// However, the situation is not so critical: there are no such loops in the
// codebase, and a Python script exists to identify potential problem areas.
//
// In the case of an `if+else` structure, the compiler requires the use of
// curly braces, giving an error at the compilation stage.
//
// In the case of a single `if`, such a place is indeed missed by the compiler,
// while the flag check is always performed, regardless of the condition, which
// does not violate the logic of operation but may lead to the execution of two
// or three extra assembly instructions.
//
//================================================================


//================================================================
//
// ERROR_HANDLING_*
//
//================================================================

#define ERROR_HANDLING_ROOT \
    bool __success = true;

#define ERROR_HANDLING_PARAMS \
    bool& __success,

#define ERROR_HANDLING_PASS \
    __success,

#define ERROR_HANDLING_MEMBER \
    bool& __success;

#define ERROR_HANDLING_CAPTURE \
    __success{__success},

#define ERROR_HANDLING_CHECK \
    ); do {if (!__success) return stdbool{};} while (0

//================================================================
//
// stdbool
//
//================================================================

struct stdbool {};

//================================================================
//
// returnTrue
// returnFalse
//
//================================================================

#define returnTrue \
    return stdbool{}

#define returnFalse \
    do {__success = false; return stdbool{};} while (0)

//================================================================
//
// requireHelper
//
//================================================================

template <typename Type>
sysinline bool requireHelper(const Type& value)
    MISSING_FUNCTION_BODY

////

template <>
sysinline bool requireHelper(const stdbool& value)
    {return true;}

////

template <>
sysinline bool requireHelper(const bool& value)
    {return value;}

//================================================================
//
// allv
//
//================================================================

sysinline stdbool allv(const stdbool& value)
    {return value;}

//================================================================
//
// require
//
//================================================================

#define require(code) \
    \
    do \
    { \
        auto __require_result = code; \
        \
        if (requireHelper(allv(__require_result))) \
            ; \
        else \
            returnFalse; \
    } \
    while (0)

//================================================================
//
// errorBlock
//
//================================================================

sysinline bool errorBlockHelper(bool& __success)
{
    bool result = __success;
    __success = true;
    return result;
}

#define errorBlock(code) \
    ((code), errorBlockHelper(__success))
