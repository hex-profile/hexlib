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
// This approach eliminates boilerplate return checks but has side effects.
// In particular, the `stdPass*` macro is no longer a single entity and splits
// the function call into two statements.
//
// As a result, if the function call is under a conditional or loop statement,
// only the first part of the macro may be executed, and the check
// will be performed after the statement.
//
// At first glance, this seems to doom this method. However, it's not so bad.
//
// In the case of an `if+else` construction, the compiler requires the use
// of curly brackets, giving an error during the compilation stage.
//
// In the case of a single `if`, such a place is indeed missed by the compiler,
// while the flag check is always performed, regardless of the condition,
// which does not disrupt the logic of work, but may lead to the execution of two
// or three extra assembly commands.
//
// To identify problematic places with loops, there is a Python script,
// the function `check_issues_in_bool_ref_error_mode`.
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
    ); do {if (!__success) return;} while (0

//================================================================
//
// returnFalse
//
//================================================================

#define returnFalse \
    do {__success = false; return;} while (0)

//================================================================
//
// require
//
//================================================================

#define require(expression) \
    \
    do \
    { \
        if (!allv(expression)) \
            returnFalse; \
    } \
    while (0)

//================================================================
//
// errorBlock
//
// Suppresses the error in the specified block of code.
//
//----------------------------------------------------------------
//
// The utility preserves a success flag, then invokes the specified function,
// subsequently restoring the original success flag state and returning a boolean
// value indicating the successful execution of the function.
//
// When starting with normal code flow, the success flag is already true.
// In this case, the function suppresses the error by maintaining the true value,
// then restoring it.
//
// However, it is not possible to simply overwrite the flag with true
// in all cases. There are scenarios where an error has already occurred,
// and the flag is set to false. During destructor execution or cleanup code,
// there might be internal cleanup functions that also need error suppression.
// It is crucial not to carelessly negate the initial error by resetting the flag
// to a true state.
//
//================================================================

template <typename Action>
sysinline bool errorBlockHelper(bool& __success, const Action& action)
{
    bool savedSuccess = __success;

    ////

    __success = true; // Required to calculate the actual success of the executed function.

    action();

    bool actionSucceeded = __success;

    ////

    __success = savedSuccess;

    return actionSucceeded;
}

//----------------------------------------------------------------

#define errorBlock(action) \
    errorBlockHelper(__success, [&] () -> void {return (action);})
