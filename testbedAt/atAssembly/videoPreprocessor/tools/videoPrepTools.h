#pragma once

#include "gpuProcessHeader.h"
#include "rndgen/rndgenBase.h"

//================================================================
//
// copyImageRect
//
//================================================================

void copyImageRect(const GpuMatrixAP<const uint8_x4>& src, const Point<Space>& ofs, const GpuMatrixAP<uint8_x4>& dst, stdPars(GpuProcessKit));

//================================================================
//
// generateGrating
//
//================================================================

void generateGrating
(
    const GpuMatrix<uint8_x4>& dst,
    const float32& period,
    const Point<float32>& transMul,
    const Point<float32>& transAdd,
    const bool& rectangleShape,
    stdPars(GpuProcessKit)
);

//================================================================
//
// generateEdge
//
//================================================================

void generateEdge
(
    const GpuMatrix<uint8_x4>& dst,
    const Point<float32>& transMul,
    const Point<float32>& transAdd,
    const float32& divSigma,
    const bool& pulse,
    stdPars(GpuProcessKit)
);

//================================================================
//
// generatePulse
//
//================================================================

void generatePulse(const GpuMatrix<uint8_x4>& dst, const Point<Space>& ofs, const Space& period, stdPars(GpuProcessKit));

//================================================================
//
// generateRandom
//
//================================================================

void generateRandom(const GpuMatrix<uint8_x4>& dst, const GpuMatrix<RndgenState>& rndgenMatrix, stdPars(GpuProcessKit));

//================================================================
//
// generateAdditionalGaussNoise
//
//================================================================

void generateAdditionalGaussNoise
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrix<uint8_x4>& dst,
    const GpuMatrix<RndgenState>& rndgenMatrix,
    const float32& sigma,
    stdPars(GpuProcessKit)
);
