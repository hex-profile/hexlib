#pragma once

#include "dataAlloc/gpuMatrixMemory.h"
#include "gpuProcessHeader.h"

namespace downsampleHalfOctave {

//================================================================
//
// DownsampleHalfOctave
//
//================================================================

class DownsampleHalfOctave
{

public:

    stdbool realloc(stdPars(GpuProcessKit));

    stdbool process(const GpuMatrix<const float16>& src, const GpuMatrix<float16>& dst, float32 dstFactor, bool testMode, stdPars(GpuProcessKit));
    stdbool process(const GpuMatrix<const float16_x2>& src, const GpuMatrix<float16_x2>& dst, float32 dstFactor, bool testMode, stdPars(GpuProcessKit));

private:

    static const Space phaseCount = 128;

    bool allocated = false;
    GpuMatrixMemory<float32> coeffs;

};

//----------------------------------------------------------------

}
