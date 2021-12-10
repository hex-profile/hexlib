#pragma once

#include "compileTools/errorHandling.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/errorLogExKit.h"

//================================================================
//
// blockExceptions
// blockExceptionsVoid
//
// Blocks all exceptions and prints them to the error log.
//
//================================================================

#define blockExceptionsVoid(action) \
    blockExceptionsVerboseHelper([&] () -> bool {action; return true;}, stdPassNoProfiling)

//----------------------------------------------------------------

#if HEXLIB_ERROR_HANDLING == 0

    #define blockExceptions(action) \
        blockExceptionsVerboseHelper([&] () -> bool {return errorBlock(action);}, stdPassNoProfiling)

#elif HEXLIB_ERROR_HANDLING == 1

    #define blockExceptions(action) \
        blockExceptionsVoid(action) // more efficient: avoid two nested catches

#else

    #error

#endif

//================================================================
//
// printExternalExceptions
//
// Should be called from a "catch" block.
//
//================================================================

void printExternalExceptions(stdPars(ErrorLogExKit)) noexcept;

//================================================================
//
// blockExceptionsVerboseHelper
//
//================================================================

template <typename Action, typename Kit>
sysinline bool blockExceptionsVerboseHelper(const Action& action, stdPars(Kit))
{
    try
    {
        return action();
    }
    catch (...) 
    {
        printExternalExceptions(stdPassThru);
        return false;
    }

    return true;
}
