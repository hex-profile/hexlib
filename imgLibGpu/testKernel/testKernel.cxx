#include "numbers/float/floatType.h"
#include "numbers/float16/float16Base.h"
#include "point/point.h"
#include "gpuSupport/gpuTexTools.h"
#include "readInterpolate/gpuTexCubic.h"

//================================================================
//
// tex2DCubicEx
//
//================================================================

template <typename SamplerType, typename CoeffsFunc>
sysinline auto tex2DCubicEx(SamplerType srcSampler, const Point<float32>& srcPos, const Point<float32>& srcTexstep, CoeffsFunc coeffsFunc)
{
    Tex2DCubicPreparation prep;
    tex2DCubicPrepare(srcPos, srcTexstep, coeffsFunc, prep);
    return tex2DCubicApply(srcSampler, prep);
}

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

//``` good
//__global__ void testKernel1(Point<float32> pos, Point<float32> texstep, float32* result)
//{
//    *result = tex2DCubicGeneric(image, pos, texstep, cubicCoeffs<float32>);
//}

////

__global__ void testKernel2(Point<float32> pos, Point<float32> texstep, float32* result)
{
    auto prep = tex2DCubicPrepare(pos, texstep, cubicCoeffs<float32>);
    *result = tex2DCubicApply(image, prep);
}

////

#endif
