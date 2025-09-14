#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// drawBackgroundPattern
//
//================================================================

void drawBackgroundPattern
(
    const Point<Space>& scrollOfs,
    const GpuMatrix<uint8_x4>& dst,
    stdPars(GpuProcessKit)
);
