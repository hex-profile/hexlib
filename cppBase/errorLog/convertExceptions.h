#pragma once

#include "compileTools/errorHandling.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/msgLogExKit.h"

//================================================================
//
// printExternalExceptions
//
// Should be called from a "catch" block.
//
//================================================================

void printExternalExceptions(stdPars(MsgLogExKit)) noexcept;

//================================================================
//
// convertExceptions
// convertExceptionsBegin
// convertExceptionsEnd
//
// Converts any exception to a hexlib error, which can be
// an exception or a boolean flag depending on hexlib error mode.
//
// Verbose: Prints the exception to the error log.
//
//================================================================

#define convertExceptionsBegin \
    try \
    {

#define convertExceptionsEndEx(errorAction) \
    } \
    catch (...) \
    { \
        printExternalExceptions(stdPassNoProfilingNc); \
        errorAction; \
    }

#define convertExceptionsEnd \
    convertExceptionsEndEx(returnFalse)

////

#define convertExceptions(action) \
    convertExceptionsBegin \
    action; \
    convertExceptionsEnd

//================================================================
//
// stdExceptBegin
// stdExceptEnd
//
//================================================================

#define stdExceptBegin \
    convertExceptionsBegin

#define stdExceptEnd \
    convertExceptionsEnd \
    return
