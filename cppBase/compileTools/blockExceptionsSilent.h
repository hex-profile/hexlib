#pragma once

#include "compileTools/errorHandling.h"

//================================================================
//
// blockExceptionsSilent
// blockExceptionsSilentVoid
// blockExceptionsSilentBool
//
// Blocks all exceptions silently.
//
//================================================================

#define blockExceptionsSilentVoid(action) \
    blockExceptionsSilentHelper([&] () -> bool {action; return true;})

#define blockExceptionsSilentBool(action) \
    blockExceptionsSilentHelper([&] () -> bool {return action;})

//----------------------------------------------------------------

#if HEXLIB_ERROR_HANDLING == 0

    #define blockExceptionsSilent(action) \
        blockExceptionsSilentHelper([&] () -> bool {return errorBlock(action);})

#elif HEXLIB_ERROR_HANDLING == 1

    #define blockExceptionsSilent(action) \
        blockExceptionsSilentVoid(action) // more efficient: avoid two nested catches

#else

    #error

#endif

//================================================================
//
// blockExceptionsSilentHelper
//
//================================================================

template <typename Action>
sysinline bool blockExceptionsSilentHelper(const Action& action)
{
    try
    {
        return action();
    }
    catch (...) 
    {
        return false;
    }

    return true;
}
