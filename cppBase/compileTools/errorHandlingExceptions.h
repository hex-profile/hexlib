#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// ERROR_HANDLING_*
//
//================================================================

#define ERROR_HANDLING_ROOT
#define ERROR_HANDLING_PARAMS
#define ERROR_HANDLING_PASS
#define ERROR_HANDLING_MEMBER
#define ERROR_HANDLING_CAPTURE
#define ERROR_HANDLING_CHECK

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

class sysnodiscard stdbool
{
};

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
// errorBlockHelper
//
//================================================================

template <typename Action>
sysinline bool errorBlockHelper(const Action& action)
{
    try
    {
        stdbool ignore = action(); // stdbool value is not used
    }
    catch (...)
    {
        return false;
    }

    return true;
}

//----------------------------------------------------------------

#define errorBlock(action) \
    errorBlockHelper([&] () -> stdbool {return (action);})
