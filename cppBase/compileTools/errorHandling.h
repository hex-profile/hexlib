//================================================================
//
// HEXLIB_ERROR_MODE
//
// 1: Using exceptions.
// 2: Using bool parameter by reference.
//
//================================================================

#ifndef HEXLIB_ERROR_MODE

    #error HEXLIB_ERROR_MODE must be defined

#endif

//================================================================
//
// Select version.
//
//================================================================

#if HEXLIB_ERROR_MODE == 0

    #error Error handling through returning bool is no longer supported after The Great Error Handling Remake.

#elif HEXLIB_ERROR_MODE == 1

    #include "errorHandlingExceptions.h"

#elif HEXLIB_ERROR_MODE == 2

    #include "errorHandlingBoolRef.h"

#else

    #error

#endif
