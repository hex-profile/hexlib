#include "errorHandling.h"

#if HEXLIB_ERROR_HANDLING == 1

//================================================================
//
// exceptThrowFailure
//
//================================================================

[[noreturn]]
void exceptThrowFailure()
{
    throw ExceptFailure();
}

//----------------------------------------------------------------

#endif
