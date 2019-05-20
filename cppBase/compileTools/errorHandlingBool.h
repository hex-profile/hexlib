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
    
#if HEXLIB_ERROR_HANDLING_EXPERIMENTAL
    explicit 
#endif
    sysinline stdbool(bool value)
        : value(value) {}

    sysinline bool getSuccessValue() const 
        {return value;}

private:

    bool value;

};

//----------------------------------------------------------------

sysinline bool allv(const stdbool& value)
    {return value.getSuccessValue();}

//================================================================
//
// returnSuccess
// returnFailure
//
//================================================================

#define returnSuccess \
    return stdbool(true)

#define returnFailure \
    return stdbool(false)

//================================================================
//
// require
//
// Check and fail without user message.
//
//================================================================

#if HEXLIB_ERROR_HANDLING_EXPERIMENTAL

    #define require(condition) \
        if (allv(condition)) ; else return stdbool(false)

#else

    #define require(condition) \
        if (allv(condition)) ; else return false

#endif

//================================================================
//
// errorBlock
//
//================================================================

#if HEXLIB_ERROR_HANDLING == 0

    sysinline bool errorBlock(const stdbool& value)
    {
        return value.getSuccessValue();
    }

#elif HEXLIB_ERROR_HANDLING == 1

    template <typename Action>
    sysinline bool exceptBlockHelper(const Action& action)
    {
        bool ok = false;

        try
        {
            action();
            ok = true;
        }
        catch (...) {}

        return ok;
    }

    #define errorBlock(action) \
        exceptBlockHelper([&] () {action;})

#endif
