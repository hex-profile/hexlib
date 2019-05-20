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
// stdvoid
//
//================================================================

using stdvoid = void;

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
// returnSuccess
// returnFailure
//
//================================================================

#define returnSuccess \
    return stdbool()

#define returnFailure \
    exceptThrowFailure()

//================================================================
//
// require
//
// Check and fail without user message.
//
//================================================================

template <typename Type>
sysinline void exceptEnsure(const Type& value);

template <>
sysinline void exceptEnsure(const stdbool& value)
{
}

template <>
sysinline void exceptEnsure(const bool& value)
{
    if_not (value)
        exceptThrowFailure();
}

//----------------------------------------------------------------

template <typename Type>
sysinline void require(const Type& value)
{
    exceptEnsure(allv(condition))
}
