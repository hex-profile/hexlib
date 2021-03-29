#if HOSTCODE
#include "visualizeComplexFilterKernel.h"
#endif

#include "gpuDevice/loadstore/loadNorm.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuSupport/gpuTool.h"
#include "imageRead/positionTools.h"
#include "mathFuncs/rotationMath.h"
#include "readInterpolate/gpuTexCubic.h"
#include "vectorTypes/vectorOperations.h"
#include "visualizeComplexFilter/types.h"

namespace visualizeComplex {

//================================================================
//
// visualizePreparationImageKernel
//
//================================================================

GPUTOOL_2D_BEG
(
    visualizePreparationImageKernel,
    ((const ComplexFloat, src, INTERP_NEAREST, BORDER_MIRROR)),
    ((ComplexFloat, dst)),
    ((Point<float32>, srcToDstFactor))
    ((Point<float32>, dstToSrcFactor))
    ((bool, interpolation))
    ((bool, dstModulation))
    ((Point<float32>, dstModulationFreq))
)
#if DEVCODE
{
    Point<float32> dstPos = point(Xs, Ys);
    Point<float32> srcPos = dstPos * dstToSrcFactor;

    auto value = zeroOf<float32_x2>();

    ////

    if_not (interpolation)
    {
        srcPos = roundPosToNearestSample(srcPos);
        dstPos = srcPos * srcToDstFactor;
    }

    if (interpolation)
        value = tex2DCubic(srcSampler, srcPos, srcTexstep);
    else
        value = tex2D(srcSampler, srcPos * srcTexstep);

    ////

    if (dstModulation)
        value = complexMul(value, circleCcw(scalarProd(dstModulationFreq, dstPos)));

    storeNorm(dst, value);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// visualizeComplexFilterFunc
//
//================================================================

#if HOSTCODE

stdbool visualizeComplexFilterFunc
(
    const GpuMatrix<const ComplexFloat>& src, 
    const GpuMatrix<ComplexFloat>& dst,
    const Point<float32>& upsampleFactor,
    bool interpolation,
    bool dstModulation,
    const Point<float32>& dstModulationFreq,
    stdPars(GpuProcessKit)
)
{
    REQUIRE(upsampleFactor > 0);
    require(visualizePreparationImageKernel(src, dst, upsampleFactor, 1.f / upsampleFactor, interpolation, dstModulation, dstModulationFreq, stdPass));

    returnTrue;
}

#endif

//----------------------------------------------------------------

}
