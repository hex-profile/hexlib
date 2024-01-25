//================================================================
//
// HEXLIB_ERROR_MODE
//
// 0: Using bool error code.
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

    #include "errorHandlingBool.h"

#elif HEXLIB_ERROR_MODE == 1

    #include "errorHandlingExceptions.h"

#elif HEXLIB_ERROR_MODE == 2

    #include "errorHandlingBoolRef.h"

#else

    #error

#endif
