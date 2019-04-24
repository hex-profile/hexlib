#include "gpuDevice/gpuDevice.h"
#include "numbers/mathIntrinsics.h"
#include "readBordered.h"
#include "data/gpuMatrix.h"
#include "vectorTypes/vectorOperations.h"
#include "mathFuncs/gaussApprox.h"
#include "rndgen/rndgenFloat.h"
#include "readInterpolate/cubicCoeffs.h"

//================================================================
//
// testKernel
//
// Place to compile and look the assembly code.
//
//================================================================

#if __CUDA_ARCH__

__global__ void testKernel(char* a, float* b, double* result)
{
}

#endif
