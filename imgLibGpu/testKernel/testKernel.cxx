#include "mathFuncs/rotationMath3d.h"
#include "numbers/float/floatType.h"
#include "vectorTypes/vectorType.h"
#include "vectorTypes/vectorOperations.h"
#include "mathFuncs/rotationMath.h"

//================================================================
//
// testKernel
//
// Place to compile and look the assembly code.
//
//================================================================

#if __CUDA_ARCH__

__global__ void testKernel_float32_x2(float32_x2* aPtr, float32_x2* bPtr, float32_x2* sumPtr)
{
    auto a = *aPtr;
    auto b = *bPtr;
    auto sum = *sumPtr;

    sum = complexFma(a, b, sum);

    *sumPtr = sum;
}

////

#endif
