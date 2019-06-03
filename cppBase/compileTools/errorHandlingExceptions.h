#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// ExceptFailure
//
// All exceptions are handled regardless of their type,
// only as an indication of failure.
//
// So the only type of internal exception is "ExceptFailure".
//
//================================================================

struct ExceptFailure {};

[[noreturn]]
void exceptThrowFailure();

//================================================================
//
// stdbool
//
//================================================================

class stdbool {};

//----------------------------------------------------------------

sysinline stdbool allv(const stdbool& value)
    {return value;}

//================================================================
//
// returnTrue
// returnFalse
//
//================================================================

#define returnTrue \
    return stdbool()

#define returnFalse \
    return (exceptThrowFailure(), stdbool())

//================================================================
//
// require
//
// Check and fail without user message.
//
//================================================================

template <typename Type>
sysinline void requireHelper(const Type& value)
    MISSING_FUNCTION_BODY

template <>
sysinline void requireHelper(const stdbool& value)
{
}

template <>
sysinline void requireHelper(const bool& value)
{
    if (!value) 
        exceptThrowFailure();
}

//----------------------------------------------------------------

#define require(value) \
    requireHelper(allv(value))

//================================================================
//
// exceptBlockHelper
//
//================================================================

template <typename Action>
sysinline bool exceptBlockHelper(const Action& action)
{
    try
    {
        action(); // stdbool value is not used
    }
    catch (...) 
    {
        return false;
    }

    return true;
}

//----------------------------------------------------------------

#define errorBlock(action) \
    exceptBlockHelper([&] () -> stdbool {return (action);})
