#pragma once

#include "circleTable/circleTable.h"
#include "gpuProcessHeader.h"

//================================================================
//
// fourierSeparable
//
//================================================================

void fourierSeparable
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

void invFourierSeparable
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

void orientedFourier
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

void invOrientedFourier
(
    const GpuMatrix<const float32_x2>& src,
    const GpuMatrix<float32_x2>& dst,
    float32 minPeriod,
    const GpuMatrix<const float32_x2>& circleTable,
    stdPars(GpuProcessKit)
);
