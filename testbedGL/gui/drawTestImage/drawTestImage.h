#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// drawTestImage
//
//================================================================

stdbool drawTestImage
(
    const Point<Space>& scrollOfs,
    int32 stripePeriodBits,
    int32 stripeWidth,
    const GpuMatrix<uint8_x4>& dst,
    stdPars(GpuProcessKit)
);
