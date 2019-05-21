//================================================================
//
// HEXLIB_ERROR_HANDLING
//
// 0: Using bool error code.
// 1: Using exceptions.
//
//================================================================

#define HEXLIB_ERROR_HANDLING 0

#define HEXLIB_ERROR_HANDLING_EXPERIMENTAL 1

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
