#if HOSTCODE
#include "downsampleHalfOctave.h"
#endif

#include "gpuDevice/loadstore/loadNorm.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/parallelLoop.h"
#include "gpuSupport/gpuTool.h"
#include "gpuSupport/gpuTexTools.h"
#include "vectorTypes/vectorOperations.h"
#include "simpleConvolutionSeparable.inl"
#include "numbers/mathIntrinsics.h"
#include "gaussSincResampling/gaussSincResamplingSettings.h"

#if HOSTCODE
#include "dataAlloc/gpuMatrixMemory.h"
#endif

namespace gaussSincResampling {
namespace downsampleHalfOctave {

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Downsample 1.414 times
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// downsampleKernel
//
//================================================================

sysinline float32 sinc(float32 x)
{
    float32 t = pi<float32> * x;
    float32 result = nativeDivide(sinf(t), t);
    if_not (def(result)) result = 1;
    return result;
}

//----------------------------------------------------------------

#define SIGMA gaussSincResampling::sigma
#define THETA gaussSincResampling::conservativeTheta

//----------------------------------------------------------------

sysinline float32 downsampleKernel(float32 x)
{
    return (1/THETA) * sinc((1/THETA) * x) * expf((-0.5f/(THETA*THETA*SIGMA*SIGMA) ) * x * x);
}

//----------------------------------------------------------------

#define DOWNSAMPLE_HALF_OCTAVE_FACTOR 1.41421356f

const Space downHoFilterLength = Space(2 * gaussSincResampling::radius * DOWNSAMPLE_HALF_OCTAVE_FACTOR + 0.999999f);

//================================================================
//
// dhoAcrossThreads
// dhoAlongThreads
//
//================================================================

static const Space dhoAcrossThreads = 16;
static const Space dhoAlongThreads = 32;

//================================================================
//
// downsampleHalfOctave.inl
//
//================================================================

#define FILTER_RADIUS gaussSincResampling::radius

#define PIXEL float16
# include "downsampleHalfOctave.inl"
#undef PIXEL

#define PIXEL float16_x2
# include "downsampleHalfOctave.inl"
#undef PIXEL

#undef FILTER_RADIUS

//================================================================
//
// makeDownsampleHalfOctaveCoeffsTransposed
//
//================================================================

GPUTOOL_2D_BEG
(
    makeDownsampleHalfOctaveCoeffsTransposed,
    PREP_EMPTY,
    ((float32, dst)),
    PREP_EMPTY
)
#if DEVCODE
{
    float32 srcOrgFrac = Xs / vGlobSize.X;

    const float32 radius = DOWNSAMPLE_HALF_OCTAVE_FACTOR * gaussSincResampling::radius;

    float32 filterBaseOfs = (-radius + 1.0f - srcOrgFrac);
    float32 coeff = downsampleKernel((filterBaseOfs + Y) * (1.f / DOWNSAMPLE_HALF_OCTAVE_FACTOR));

    ////

    float32 sum = 0;

    for (Space i = 0; i < vGlobSize.Y; ++i)
        sum += downsampleKernel((filterBaseOfs + i) * (1.f / DOWNSAMPLE_HALF_OCTAVE_FACTOR));

    ////

    storeNorm(dst, coeff / sum);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// makeDownsampleHalfOctaveCoeffsNormal
//
//================================================================

GPUTOOL_2D_BEG
(
    makeDownsampleHalfOctaveCoeffsNormal,
    PREP_EMPTY,
    ((float32, dst)),
    PREP_EMPTY
)
#if DEVCODE
{
    float32 srcOrgFrac = Ys / vGlobSize.Y;

    const float32 radius = DOWNSAMPLE_HALF_OCTAVE_FACTOR * gaussSincResampling::radius;

    float32 filterBaseOfs = (-radius + 1.0f - srcOrgFrac);
    float32 coeff = downsampleKernel((filterBaseOfs + X) * (1.f / DOWNSAMPLE_HALF_OCTAVE_FACTOR));

    ////

    float32 sum = 0;

    for (Space i = 0; i < vGlobSize.X; ++i)
        sum += downsampleKernel((filterBaseOfs + i) * (1.f / DOWNSAMPLE_HALF_OCTAVE_FACTOR));

    ////

    storeNorm(dst, coeff / sum);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// DownsampleHalfOctaveConservative::realloc
//
//================================================================

#if HOSTCODE 

stdbool DownsampleHalfOctaveConservative::realloc(stdPars(GpuProcessKit))
{
    stdBegin;

    allocated = false;

    require(coeffs.realloc(point(downHoFilterLength, phaseCount), stdPass));
    require(makeDownsampleHalfOctaveCoeffsNormal(coeffs, stdPass));

    allocated = true;

    stdEnd;
}

#endif

//----------------------------------------------------------------

}
}
