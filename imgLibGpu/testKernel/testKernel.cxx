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
    *value = __fdividef(1, *value);
}

////

#endif
