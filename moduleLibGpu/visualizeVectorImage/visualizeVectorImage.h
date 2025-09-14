#pragma once

#include "gpuProcessHeader.h"
#include "types/lt/ltBase.h"
#include "imageRead/interpType.h"
#include "imageRead/borderMode.h"

//================================================================
//
// visualizeVectorImage
//
//================================================================

template <typename VectorType>
void visualizeVectorImage
(
    const GpuMatrix<const VectorType>& src,
    const GpuMatrixAP<uint8_x4>& dst,
    const LinearTransform<Point<float32>>& coordBackTransform,
    float32 vectorFactor,
    InterpType interpType,
    BorderMode borderMode,
    bool grayMode,
    stdPars(GpuProcessKit)
);

//================================================================
//
// imposeVectorArrow
//
//================================================================

void imposeVectorArrow
(
    const GpuMatrixAP<uint8_x4>& dst,
    const Point<float32>& vectorBegin,
    const Point<float32>& vectorValue,
    const bool& orientationMode,
    stdPars(GpuProcessKit)
);
