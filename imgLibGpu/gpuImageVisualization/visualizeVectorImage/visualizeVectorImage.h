#pragma once

#include "gpuProcessHeader.h"
#include "numbers/lt/ltBase.h"
#include "imageRead/interpType.h"
#include "imageRead/borderMode.h"

//================================================================
//
// visualizeVectorImage
//
//================================================================

template <typename VectorType>
bool visualizeVectorImage
(
    const GpuMatrix<const VectorType>& src,
    const GpuMatrix<uint8_x4>& dst,
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

bool imposeVectorArrow
(
    const GpuMatrix<uint8_x4>& dst,
    const Point<float32>& vectorBegin,
    const Point<float32>& vectorValue,
    stdPars(GpuProcessKit)
);
