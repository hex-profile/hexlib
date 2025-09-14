#include "exceptionTestAux.h"

#include <stdlib.h>

namespace exceptionTest {

//================================================================
//
// myMalloc
// myFree
//
//================================================================

void* myMalloc(int n)
{
    return nullptr;
}

void myFree(void* ptr)
{
}

//================================================================
//
// throwFailure
//
//================================================================

[[noreturn]]
void throwFailure()
{
    throw Failure{};
}

//================================================================
//
// newApproachCallee
//
//================================================================

void newApproachCallee(bool& successFlag, stdPars(DiagnosticKit))
{
}

//----------------------------------------------------------------

}
