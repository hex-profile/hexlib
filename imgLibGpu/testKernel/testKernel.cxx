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

__global__ void testKernel1(float32* value)
{
    *value = fastSqrt(*value);
}

////

__global__ void testKernel2(float32* value)
{
    *value = sqrtf(*value);
}

////

#endif
