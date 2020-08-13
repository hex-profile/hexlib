#pragma once

#include "compileTools/errorHandling.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/errorLogExKit.h"

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
// convertAllExceptionsHelper
//
//================================================================

template <typename Action, typename Kit>
sysinline stdbool convertAllExceptionsHelper(const Action& action, stdPars(Kit))
{
    try
    {
        return action();
    }
    catch (...) 
    {
        printExternalExceptions(stdPassThru);
        returnFalse;
    }

    returnTrue;
}

//----------------------------------------------------------------

#define convertAllExceptions(action) \
    convertAllExceptionsHelper([&] () -> stdbool {return action;}, stdPass)
