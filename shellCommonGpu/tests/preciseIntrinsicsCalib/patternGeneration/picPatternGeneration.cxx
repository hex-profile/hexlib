#include "picPatternGeneration.h"

#include "rndgen/rndgenFloat.h"
#include "rndgen/rndgenMix.h"
#include "imageRead/positionTools.h"
#include "rndgen/randRange.h"
#include "gpuSupport/reductionTool/reductionToolModern.h"
#include "gpuSupport/gpuTool/gptGetFlatProcessor.h"
#include "gpuDevice/gpuAtomics.h"

namespace picPatternGeneration {

//================================================================
//
// simpleNoise2D
//
// https://www.shadertoy.com/view/MdcfDj
//
//================================================================

sysinline uint32 simpleNoise2D(uint32 X, uint32 Y)
{
    constexpr uint32 M1 = 929 * 1719413u;
    constexpr uint32 M2 = 11 * 2467 * 140473u;

    X *= M1;
    Y *= M2;

    return (X ^ Y) * M1;
}

//================================================================
//
// genRandomMatrix
//
//================================================================

GPUTOOL_2D_BEG
(
    genRandomMatrix,
    PREP_EMPTY,
    ((float32, dst)),
    ((uint32, seed))
)
#if DEVCODE
{
    RndgenState rndgen = seed ^ simpleNoise2D(X ^ 0xAA090ED5, Y ^ 0x9B1018C9);
    auto randomValue = rndgenUniformSignedFloat(rndgen, 2);
    *dst = randomValue;
}
#endif
GPUTOOL_2D_END

//================================================================
//
// combineLinearly
//
//================================================================

GPUTOOL_2D_BEG
(
    combineLinearly,
    PREP_EMPTY,
    ((float32, src1))
    ((float32, src2))
    ((float32, dst)),
    ((float32, c1))
    ((float32, c2))
)
#if DEVCODE
{
    *dst = c1 * helpRead(*src1) + c2 * helpRead(*src2);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// computeStats
//
//================================================================

GPUTOOL_2D_BEG_EX
(
    computeStats,
    GPUTOOL_2D_DEFAULT_THREAD_COUNT,
    true,
    PREP_EMPTY,
    ((const float32, src)),
    ((GpuArray<float32>, dstSumAbs))
    ((GpuArray<float32>, dstSumSq))
)
#if DEVCODE
{
    auto value = vItemIsActive ? *src : float32{0};

    auto sumAbs = absv(value);
    auto sumSq = square(value);

    ////

    GPT_GET_FLAT_PROCESSOR_2D;

    ////

    REDUCTION_MODERN_MAKE
    (
        reduct,
        threadCount, threadIndex,
        ((float32, sumAbs, 0.f))
        ((float32, sumSq, 0.f)),
        {
            *sumAbsL += *sumAbsR;
            *sumSqL += *sumSqR;
        }
    );

    ////

    if (threadIsMain)
    {
        devAbortCheck(dstSumAbs.size() == 1);
        devAbortCheck(dstSumSq.size() == 1);

        atomicAdd(&dstSumAbs[0], sumAbs);
        atomicAdd(&dstSumSq[0], sumSq);
    }
}
#endif
GPUTOOL_2D_END


//----------------------------------------------------------------

}
