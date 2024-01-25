#include "errorHandling.h"

#if HEXLIB_ERROR_MODE == 1

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
