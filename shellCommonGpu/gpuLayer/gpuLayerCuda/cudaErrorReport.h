#pragma once

#include <cuda.h>

#include "stdFunc/stdFunc.h"
#include "errorLog/debugBreak.h"
#include "extLib/userOutput/msgKind.h"

//================================================================
//
// REQUIRE_CUDA
// CHECK_CUDA
//
//================================================================

template <typename Kit>
sysinline stdbool checkCudaHelper(CUresult cudaErr, const CharType* statement, stdPars(Kit))
{
    if (cudaErr == CUDA_SUCCESS)
        returnTrue;

    require(printMsgTrace(STR("CUDA error: %0: %1."), statement, cudaErr, msgErr, stdPassThru));
    returnFalse;
}

//----------------------------------------------------------------

#define REQUIRE_CUDA(statement) \
    require(checkCudaHelper(statement, PREP_STRINGIZE(statement), stdPass))

#define DEBUG_BREAK_CHECK_CUDA(statement) \
    DEBUG_BREAK_CHECK((statement) == CUDA_SUCCESS)
