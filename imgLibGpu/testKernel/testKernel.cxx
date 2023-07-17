#include "mathFuncs/rotationMath3d.h"
#include "numbers/float/floatType.h"
#include "vectorTypes/vectorType.h"
#include "vectorTypes/vectorOperations.h"
#include "mathFuncs/rotationMath.h"
#include "imageRead/positionTools.h"

//================================================================
//
// testKernel
//
// Place to compile and look the assembly code.
//
//================================================================

#if __CUDA_ARCH__

__global__ void testKernel(Space* ptr, Space maxSize)
{
    auto value = *ptr;

    value = clampRange(value, 0, maxSize);

    *ptr = value;
}

////

#endif
