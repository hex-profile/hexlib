#pragma once

#include "compileTools/compileTools.h"

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

class stdbool 
#if HEXLIB_PLATFORM == 0
    [[nodiscard]]
#endif
{

public:
    
    sysinline explicit stdbool(bool value)
        : value(value) {}

    sysinline bool getSuccessValue() const 
        {return value;}

private:

    bool value;

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
    return stdbool(true)

#define returnFalse \
    return stdbool(false)

//================================================================
//
// require
//
// Check and fail without user message.
//
//================================================================

template <typename Type>
sysinline bool requireHelper(const Type& value);

template <>
sysinline bool requireHelper(const stdbool& value)
    {return value.getSuccessValue();}

template <>
sysinline bool requireHelper(const bool& value)
    {return value;}

//----------------------------------------------------------------

#define require(condition) \
    if (requireHelper(allv(condition))) ; else return stdbool(false)

//================================================================
//
// errorBlock
//
//================================================================

sysinline bool errorBlock(const stdbool& value)
{
    return value.getSuccessValue();
}
