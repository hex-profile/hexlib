//----------------------------------------------------------------
//
//
//
//----------------------------------------------------------------
#include "downsampleTwiceModel.h"

#include "gpuDevice/gpuDevice.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "kit/kit.h"
#include "data/gpuMatrix.h"
#include "gpuSupport/gpuTemplateKernel.h"
#include "gpuSupport/gpuTool.h"

#if HOSTCODE
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "numbers/divRound.h"
#include "dataAlloc/gpuMatrixMemory.h"
#endif

namespace downsampleTwiceModelSpace {

//================================================================
//
// Filter coeffs
//
//================================================================

#if DEVCODE

devConstant float32 filterCoeffs[] = 
{
    +0.00019185f, +0.00072782f, +0.00238092f, +0.00671621f, +0.01633667f, +0.03426598f, +0.06197591f, +0.09665909f, +0.12999377f, +0.15075178f, +0.15075178f, +0.12999377f, +0.09665909f, +0.06197591f, +0.03426598f, +0.01633667f, +0.00671621f, +0.00238092f, +0.00072782f, +0.00019185f
};

static const Space filterSize = COMPILE_ARRAY_SIZE(filterCoeffs);
COMPILE_ASSERT(filterSize % 2 == 0);

#endif

//================================================================
//
// separableDownsampleTwice
//
//================================================================

GPUTOOL_2D_BEG
(
    separableDownsampleTwice,
    ((const Type, src, INTERP_NONE, BORDER_MIRROR)),
    ((Type, dst)),
    ((bool, horizontal))
)

#if DEVCODE
{
    float32 sum = 0;

    Space tX = X;
    Space tY = Y;

    (horizontal ? tX : tY) *= 2;
    (horizontal ? tX : tY) -= (filterSize/2 - 1);

    for (Space k = 0; k < filterSize; ++k)
    {
        float32 value = devTex2D(srcSampler, (tX + 0.5f) * srcTexstep.X, (tY + 0.5f) * srcTexstep.Y);
        sum += filterCoeffs[k] * value;
        (horizontal ? tX : tY)++;
    }

    storeNorm(dst, sum);
}
#endif

GPUTOOL_2D_END

//================================================================
//
// downsampleTwiceModel
//
//================================================================

#if HOSTCODE

bool downsampleTwiceModel(const GpuMatrix<const Type>& src, const GpuMatrix<Type>& dst, stdPars(GpuProcessKit))
{
    stdBegin;

    ////

    GPU_MATRIX_ALLOC(tmp, Type, point(src.sizeX(), dst.sizeY()));
    require(separableDownsampleTwice(src, tmp, false, stdPass));
    require(separableDownsampleTwice(tmp, dst, true, stdPass));

    ////

    stdEnd;
}

//----------------------------------------------------------------

#endif

}
