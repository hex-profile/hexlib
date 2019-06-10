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

__global__ void testKernel(const Point4D<float32>* Q, const Point3D<float32>* V, Point3D<float32>* result)
{
    *result = quatRotateVec(*Q, *V);
}

#endif
