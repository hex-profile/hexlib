#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// HEXLIB_ERROR_HANDLING
//
// 0: Using boolean error codes.
// 1: Using exceptions.
//
//================================================================

#define HEXLIB_ERROR_HANDLING 0

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

#if HEXLIB_ERROR_HANDLING == 1

[[noreturn]]
void exceptThrowFailure();

#endif

//================================================================
//
// stdbool
// stdvoid
//
//================================================================

using stdvoid = void;

//----------------------------------------------------------------

#if HEXLIB_ERROR_HANDLING == 0

    class stdbool 
    #if HEXLIB_PLATFORM == 0
        [[nodiscard]]
    #endif
    {

    public:
    
        sysinline stdbool(bool value)
            : value(value) {}

        sysinline bool getSuccessValue() const 
            {return value;}

    private:

        bool value;

    };

    sysinline bool allv(const stdbool& value)
        {return value.getSuccessValue();}

#elif HEXLIB_ERROR_HANDLING == 1

    class stdbool {};

    sysinline stdbool allv(const stdbool& value)
        {return value;}

#endif

//================================================================
//
// returnSuccess
//
//================================================================

#if HEXLIB_ERROR_HANDLING == 0

    #define returnSuccess \
        return stdbool(true)

    #define returnFailure \
        return stdbool(false)

#elif HEXLIB_ERROR_HANDLING == 1

    #define returnSuccess \
        return stdbool()

    #define returnFailure \
        exceptThrowFailure()

#endif

//================================================================
//
// exceptEnsure
//
// Check and fail without user message.
//
//================================================================

#if HEXLIB_ERROR_HANDLING == 1

template <typename Type>
sysinline void exceptEnsure(const Type& value);

template <>
sysinline void exceptEnsure(const stdbool& value)
{
}

template <>
sysinline void exceptEnsure(const bool& value)
{
    if (!value)
        exceptThrowFailure();
}

#endif

//================================================================
//
// require
//
// Check for a function call fail.
//
//================================================================

#if HEXLIB_ERROR_HANDLING == 0

    #define require(condition) \
        if (allv(condition)) ; else return false

#elif HEXLIB_ERROR_HANDLING == 1
   
    #define require(condition) \
        exceptEnsure(allv(condition))

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
