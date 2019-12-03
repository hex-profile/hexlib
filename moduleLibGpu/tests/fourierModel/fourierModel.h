#pragma once

#include "circleTable/circleTable.h"
#include "gpuProcessHeader.h"

//================================================================
//
// fourierSeparable
//
//================================================================

stdbool fourierSeparable
(
    const GpuMatrix<const float32_x2>& src,
    const GpuMatrix<float32_x2>& dst,
    const Point<float32>& minPeriod,
    const GpuMatrix<const float32_x2>& circleTable,
    stdPars(GpuProcessKit)
);

//================================================================
//
// invFourierSeparable
//
//================================================================

stdbool invFourierSeparable
(
    const GpuMatrix<const float32_x2>& src,
    const GpuMatrix<float32_x2>& dst,
    const Point<float32>& minPeriod,
    const GpuMatrix<const float32_x2>& circleTable,
    bool normalize,
    stdPars(GpuProcessKit)
);

//================================================================
//
// orientedFourier
//
//================================================================

stdbool orientedFourier
(
    const GpuMatrix<const float32_x2>& src,
    const GpuMatrix<float32_x2>& dst,
    float32 minPeriod,
    const GpuMatrix<const float32_x2>& circleTable,
    stdPars(GpuProcessKit)
);

//================================================================
//
// invOrientedFourier
//
//================================================================

stdbool invOrientedFourier
(
    const GpuMatrix<const float32_x2>& src,
    const GpuMatrix<float32_x2>& dst,
    float32 minPeriod,
    const GpuMatrix<const float32_x2>& circleTable,
    stdPars(GpuProcessKit)
);
