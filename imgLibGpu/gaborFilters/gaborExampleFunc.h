#pragma once

#include "gpuProcessHeader.h"
#include "numbers/float16/float16Base.h"
#include "data/gpuLayeredMatrix.h"

//================================================================
//
// gaborExampleFunc
//
//================================================================

stdbool gaborExampleFunc
(
    const GpuMatrix<const float16>& src,
    const GpuMatrix<const float32_x2>& circleTable,
    const GpuLayeredMatrix<float16_x2>& dst,
    bool demodulateOutput,
    bool horizontallyFirst,
    bool uncachedVersion,
    stdPars(GpuProcessKit)
);
