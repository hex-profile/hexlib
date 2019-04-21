#include "upsampleTwiceModel.h"

#if HOSTCODE
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "numbers/divRound.h"
#include "errorLog/errorLog.h"
#include "dataAlloc/gpuMatrixMemory.h"
#endif

#include "gpuDevice/gpuDevice.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "kit/kit.h"
#include "data/gpuMatrix.h"
#include "gpuSupport/gpuTool.h"

namespace upsampleTwiceModelSpace {

//================================================================
//
// Filter coeffs
//
//================================================================

#if DEVCODE

devConstant float32 filterCoeffs[] = 
{
    -0.00029563f, -0.00097728f, +0.00179037f, +0.00274977f, -0.00387311f, -0.00518189f, +0.00670312f, +0.00847177f, -0.01053452f, -0.01295565f, +0.01582671f, +0.01928290f, -0.02353269f, -0.02891464f, +0.03601628f, +0.04595134f, -0.06111218f, -0.08772803f, +0.14865567f, +0.44965770f, +0.44965770f, +0.14865567f, -0.08772803f, -0.06111218f, +0.04595134f, +0.03601628f, -0.02891464f, -0.02353269f, +0.01928290f, +0.01582671f, -0.01295565f, -0.01053452f, +0.00847177f, +0.00670312f, -0.00518189f, -0.00387311f, +0.00274977f, +0.00179037f, -0.00097728f, -0.00029563f
};

static const Space filterSize = COMPILE_ARRAY_SIZE(filterCoeffs);
COMPILE_ASSERT(filterSize % 2 == 0);

#endif

//================================================================
//
// separableUpsampleTwice
//
//================================================================

GPUTOOL_2D_BEG
(
    separableUpsampleTwice,
    ((const Type, src, INTERP_NONE, BORDER_CLAMP)),
    ((Type, dst)),
    ((bool, horizontal))
)

#if DEVCODE
{
    float32 sum = 0;

    Space tX = X;
    Space tY = Y;

    (horizontal ? tX : tY) -= (filterSize/2 + 1);
    Space p = ~((horizontal ? tX : tY) & 1);

    (horizontal ? tX : tY) >>= 1;

    for (Space k = p; k < filterSize; k += 2)
    {
        float32 value = devTex2D(srcSampler, (tX + 0.5f) * srcTexstep.X, (tY + 0.5f) * srcTexstep.Y);
        sum += 2 * filterCoeffs[k] * value;
        (horizontal ? tX : tY)++;
    }

    storeNorm(dst, sum);
}
#endif

GPUTOOL_2D_END

//================================================================
//
// upsampleTwiceModel
//
//================================================================

#if HOSTCODE

stdbool upsampleTwiceModel(const GpuMatrix<const Type>& src, const GpuMatrix<Type>& dst, stdPars(GpuProcessKit))
{
    stdBegin;

    ////

    GPU_MATRIX_ALLOC(tmp, Type, point(src.sizeX(), dst.sizeY()));
    require(separableUpsampleTwice(src, tmp, false, stdPass));
    require(separableUpsampleTwice(tmp, dst, true, stdPass));

    ////

    stdEnd;
}

#endif

//----------------------------------------------------------------

}
