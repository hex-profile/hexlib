#include "compileTools.h"

//================================================================
//
// throwFailure
//
//================================================================

[[noreturn]]
void throwFailure()
{
    throw Failure();
}
