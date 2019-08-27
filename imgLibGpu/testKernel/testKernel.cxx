#include "numbers/float/floatType.h"
#include "numbers/float16/float16Base.h"
#include "point/point.h"
#include "gpuSupport/gpuTexTools.h"
#include "readInterpolate/gpuTexCubic.h"

//================================================================
//
// testKernel
//
// Place to compile and look the assembly code.
//
//================================================================

#if __CUDA_ARCH__

devDefineSampler(image, DevSampler2D, DevSamplerFloat, 1)

////

__global__ void testKernel1(Point<float32> pos, Point<float32> texstep, float32* result)
{
    *result = tex2DCubicGeneric<CubicCoeffs>(image, pos, texstep);
}

////

__global__ void testKernel2(Point<float32> pos, Point<float32> texstep, float32* result)
{
    auto prep = tex2DCubicPrepare<CubicCoeffs>(pos, texstep);
    *result = tex2DCubicApply(image, prep);
}

////

#endif
