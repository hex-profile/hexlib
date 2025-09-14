#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// ERROR_HANDLING_*
//
//================================================================

#define ERROR_HANDLING_ROOT \
    bool __ok = true;

#define ERROR_HANDLING_PARAMS \
    bool& __ok,

#define ERROR_HANDLING_PASS \
    __ok,

#define ERROR_HANDLING_MEMBER \
    bool& __ok;

#define ERROR_HANDLING_CAPTURE \
    __ok{__ok},

//================================================================
//
// HEXLIB_STDBOOL_CLASS
//
//================================================================

#ifdef _DEBUG
    #define HEXLIB_STDBOOL_CLASS 1
#else
    #define HEXLIB_STDBOOL_CLASS 0
#endif

//================================================================
//
// stdbool
//
//================================================================

#if !HEXLIB_STDBOOL_CLASS

    using stdbool = bool;

#else

    class sysnodiscard stdbool
    {

    public:

        sysinline explicit stdbool(bool value)
            : value(value) {}

        sysinline bool getSuccessValue() const
            {return value;}

    private:

        bool value;

    };

    ////

    sysinline stdbool allv(const stdbool& value)
        {return value;}

#endif


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

////

#if HEXLIB_STDBOOL_CLASS

    template <>
    sysinline bool requireHelper(const stdbool& value)
        {return value.getSuccessValue();}

#endif

////

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

#if !HEXLIB_STDBOOL_CLASS
    return value;
#else
    return value.getSuccessValue();
#endif

}
