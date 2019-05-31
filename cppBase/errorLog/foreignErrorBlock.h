#pragma once

#include "compileTools/errorHandling.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/errorLogExKit.h"

//================================================================
//
// reportForeignException
//
//================================================================

void reportForeignException(stdPars(ErrorLogExKit)) noexcept;

//================================================================
//
// foreignExceptReport*
//
//================================================================

template <typename Action, typename Kit>
sysinline bool foreignErrorBlockHelper(const Action& action, stdPars(Kit))
{
    try
    {
    #if HEXLIB_ERROR_HANDLING == 0
        return errorBlock(action());
    #elif HEXLIB_ERROR_HANDLING == 1
        action();
    #else
        #error
    #endif
    }
    catch (...) 
    {
        reportForeignException(stdPassThru);
        return false;
    }

    return true;
}

//----------------------------------------------------------------

#define foreignErrorBlock(action) \
    foreignErrorBlockHelper([&] () -> stdbool {return (action);}, stdPass)
