#pragma once

#include "compileTools/errorHandling.h"

//================================================================
//
// blockExceptBegin
// blockExceptEnd
//
// Silently catches any exception and makes a specified action.
//
//================================================================

#define blockExceptBegin \
    try \
    {

#define blockExceptEnd(errorAction) \
    } \
    catch (...) \
    { \
        errorAction; \
    }

#define blockExceptEndIgnore \
    blockExceptEnd({})

//----------------------------------------------------------------

#define boolFuncExceptBegin \
    blockExceptBegin

#define boolFuncExceptEnd \
    blockExceptEnd(return false); \
    return true
