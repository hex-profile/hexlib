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

#define foreignExceptReportBeg \
    \
    try {

#define foreignExceptReportEnd \
    \
    } \
    catch (...) \
    { \
        reportForeignException(stdPass); \
        exceptThrowFailure(); \
    }
