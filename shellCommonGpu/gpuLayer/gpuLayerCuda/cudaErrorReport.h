#pragma once

#include <cuda.h>

#include "stdFunc/stdFunc.h"
#include "errorLog/debugBreak.h"

//================================================================
//
// REQUIRE_CUDA
// CHECK_CUDA
//
//================================================================

template <typename Kit>
inline bool checkCudaHelper(CUresult cudaErr, const CharType* statement, stdPars(Kit))
{
    if (cudaErr == CUDA_SUCCESS)
        return true;

    printMsgTrace(kit.errorLogEx, STR("CUDA error: %0 returned %1"), statement, cudaErr, msgErr, stdPassThru);
    return false;
}

//----------------------------------------------------------------

#define CHECK_CUDA(statement) \
    checkCudaHelper(statement, PREP_STRINGIZE(statement), stdPass)

#define REQUIRE_CUDA(statement) \
    require(CHECK_CUDA(statement))

#define DEBUG_BREAK_CHECK_CUDA(statement) \
    DEBUG_BREAK_CHECK((statement) == CUDA_SUCCESS)

