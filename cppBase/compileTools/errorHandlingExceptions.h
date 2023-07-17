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

class
#if (__cplusplus >= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L) && (_MSC_VER >= 1913))
    [[nodiscard]]
#endif
stdbool
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
