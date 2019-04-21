#pragma once

#include "gpuProcessHeader.h"
#include "rndgen/rndgenBase.h"

//================================================================
//
// copyImageRect
//
//================================================================

stdbool copyImageRect(const GpuMatrix<const uint8_x4>& src, const Point<Space>& ofs, const GpuMatrix<uint8_x4>& dst, stdPars(GpuProcessKit));

//================================================================
//
// generateGrating
//
//================================================================

stdbool generateGrating
(
    const GpuMatrix<uint8_x4>& dst,
    const float32& period,
    const Point<float32>& transMul, const Point<float32>& transAdd,
    const bool& rectangleShape,
    stdPars(GpuProcessKit)
);

//================================================================
//
// generatePulse
//
//================================================================

stdbool generatePulse(const GpuMatrix<uint8_x4>& dst, const Point<Space>& ofs, const Space& period, stdPars(GpuProcessKit));

//================================================================
//
// generateRandom
//
//================================================================

stdbool generateRandom(const GpuMatrix<uint8_x4>& dst, const GpuMatrix<RndgenState>& rndgenMatrix, stdPars(GpuProcessKit));

//================================================================
//
// generateAdditionalGaussNoise
//
//================================================================

stdbool generateAdditionalGaussNoise
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrix<uint8_x4>& dst,
    const GpuMatrix<RndgenState>& rndgenMatrix,
    const float32& sigma,
    stdPars(GpuProcessKit)
);
