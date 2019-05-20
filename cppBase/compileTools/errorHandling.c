#include "errorHandling.h"

//================================================================
//
// exceptThrowFailure
//
//================================================================

#if HEXLIB_ERROR_HANDLING == 1

[[noreturn]]
void exceptThrowFailure()
{
    throw ExceptFailure();
}

#endif
