#include "errorHandlingBase.h"

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

//================================================================
//
//
//
//================================================================

void test()
{
    exceptBlockBegEx(ok)
    
    throwFailure();

    exceptBlockEndEx;
    



}

