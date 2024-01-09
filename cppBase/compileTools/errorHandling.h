//================================================================
//
// HEXLIB_ERROR_HANDLING
//
// 0: Using bool error code.
// 1: Using exceptions.
// 2: Using bool parameter by reference.
//
//================================================================

#ifndef HEXLIB_ERROR_HANDLING

    #define HEXLIB_ERROR_HANDLING 0

#endif

//================================================================
//
// Select version.
//
//================================================================

#if HEXLIB_ERROR_HANDLING == 0

    #include "errorHandlingBool.h"

#elif HEXLIB_ERROR_HANDLING == 1

    #include "errorHandlingExceptions.h"

#elif HEXLIB_ERROR_HANDLING == 2

    #include "errorHandlingBoolRef.h"

#else

    #error

#endif
