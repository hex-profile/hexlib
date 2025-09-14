#pragma once

#include "gpuProcessHeader.h"
#include "gpuSupport/gpuTool.h"

namespace picPatternGeneration {

//================================================================
//
// genRandomMatrix
//
//================================================================

GPUTOOL_2D_PROTO
(
    genRandomMatrix,
    PREP_EMPTY,
    ((float32, dst)),
    ((uint32, seed))
);

//================================================================
//
// combineLinearly
//
//================================================================

GPUTOOL_2D_PROTO
(
    combineLinearly,
    PREP_EMPTY,
    ((float32, src1))
    ((float32, src2))
    ((float32, dst)),
    ((float32, c1))
    ((float32, c2))
);

//================================================================
//
// computeStats
//
//================================================================

GPUTOOL_2D_PROTO
(
    computeStats,
    PREP_EMPTY,
    ((const float32, src)),
    ((GpuArray<float32>, dstSumAbs))
    ((GpuArray<float32>, dstSumSq))
);

//----------------------------------------------------------------

}
