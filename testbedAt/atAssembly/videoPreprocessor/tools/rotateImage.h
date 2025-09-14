#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// centerMatchedSpaceRotation
//
// angle in [0, 1) range
//
// SPACE coordinates are assumed: pixels at 0.5, 1.5, etc.
//
// Derivation:
// srcCenter = dstCenter * rotation + shift
// shift = srcCenter - dstCenter * rotation
//
//================================================================

template <typename Float>
sysinline void centerMatchedSpaceRotation
(
    const Point<Space>& srcSize,
    const Point<Space>& dstSize,
    const Point<Float>& rotation,
    Point<Float>& transMul, Point<Float>& transAdd
)
{
    Point<Float> srcCenter = convertNearest<Float>(srcSize) * 0.5f;
    Point<Float> dstCenter = convertNearest<Float>(dstSize) * 0.5f;

    Point<Float> shift = srcCenter - complexMul(dstCenter, rotation);

    transMul = rotation;
    transAdd = shift;
}

//================================================================
//
// rotateImageLinearZero
//
//================================================================

void rotateImageLinearZero
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrixAP<uint8_x4>& dst,
    const Point<float32>& transMul, const Point<float32>& transAdd,
    stdPars(GpuProcessKit)
);

//================================================================
//
// rotateImageLinearMirror
//
//================================================================

void rotateImageLinearMirror
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrixAP<uint8_x4>& dst,
    const Point<float32>& transMul, const Point<float32>& transAdd,
    stdPars(GpuProcessKit)
);

//================================================================
//
// rotateImageCubicZero
//
//================================================================

void rotateImageCubicZero
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrixAP<uint8_x4>& dst,
    const Point<float32>& transMul, const Point<float32>& transAdd,
    stdPars(GpuProcessKit)
);

//================================================================
//
// rotateImageCubicMirror
//
//================================================================

void rotateImageCubicMirror
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrixAP<uint8_x4>& dst,
    const Point<float32>& transMul, const Point<float32>& transAdd,
    stdPars(GpuProcessKit)
);
