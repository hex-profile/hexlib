#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// convertColorPixelToMonoPixel
//
//================================================================

stdbool convertColorPixelToMonoPixel
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrix<uint8>& dst,
    stdPars(GpuProcessKit)
);
