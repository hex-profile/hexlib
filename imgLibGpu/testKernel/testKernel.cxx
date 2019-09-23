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

__global__ void testKernelQuat(Point3D<float32>* vec, Point4D<float32> R)
{
    *vec = ~R % (*vec);
}

////

__global__ void testKernelMat(Point3D<float32>* vec, Mat3D<float32> R)
{
    *vec = ~R % (*vec);
}

////

#endif
