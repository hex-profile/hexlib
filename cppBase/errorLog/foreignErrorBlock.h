#pragma once

#include "userOutput/errorLogExKit.h"
#include "stdFunc/stdFunc.h"

#include "compileTools/errorHandling.h"

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
// Any foreign exception is reported to error log 
// and re-thrown as a native "ExceptFailure".
//
//================================================================

template <typename Action, typename Kit>
sysinline bool foreignErrorBlockHelper(const Action& action, const Kit& kit)
{
    bool ok = false;

    try
    {
        action();
        ok = true;
    }
    catch (...) 
    {
        reportForeignException(stdPass);
    }

    return ok;
}

//----------------------------------------------------------------

#define foreignErrorBlock(action) \
    errorBlock(foreignErrorBlockHelper([&] () {action;}, kit))
