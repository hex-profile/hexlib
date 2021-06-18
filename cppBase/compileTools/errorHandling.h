//================================================================
//
// HEXLIB_ERROR_HANDLING
//
// 0: Using bool error code.
// 1: Using exceptions.
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

#else

    #error

#endif
