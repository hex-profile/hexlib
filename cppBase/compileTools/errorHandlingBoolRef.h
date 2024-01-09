#pragma once

#include "compileTools/compileTools.h"

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
