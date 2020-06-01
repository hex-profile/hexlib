#include "mathFuncs/rotationMath3d.h"
#include "numbers/float/floatType.h"

//================================================================
//
// testKernel
//
// Place to compile and look the assembly code.
//
//================================================================

#if __CUDA_ARCH__

__global__ void testKernel1(float32* ptr)
{
    auto value = *ptr;

    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFF, value, 16));
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFF, value, 8));
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFF, value, 4));
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFF, value, 2));
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFF, value, 1));

    *ptr = value;
}

////

#endif
