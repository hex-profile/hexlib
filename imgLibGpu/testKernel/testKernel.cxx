#include "mathFuncs/rotationMath3d.h"
#include "numbers/float/floatType.h"
#include "vectorTypes/vectorType.h"
#include "vectorTypes/vectorOperations.h"
#include "mathFuncs/rotationMath.h"
#include "imageRead/positionTools.h"
#include "data/gpuArray.h"
#include "gpuDevice/gpuDevice.h"
#include "data/gpuMatrix.h"

//================================================================
//
// testKernel
//
// Place to compile and look the assembly code.
//
//================================================================

#if __CUDA_ARCH__

__global__ void testKernel(GpuMatrix<Point3D<int32>> mat, Space X, Space Y)
{
    devAbortCheck(mat.element(X, Y) != 0);
}

#endif
