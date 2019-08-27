#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"
#include "imageRead/interpType.h"
#include "types/lt/ltBase.h"

//================================================================
//
// warpImage
//
// Image interpolation and border mode are specified.
// Map parameters are always INTERP_LINEAR and BORDER_CLAMP.
//
//================================================================

template <typename SrcPixel, typename DstPixel>
stdbool warpImage
(
    const GpuMatrix<const SrcPixel>& src,
    const LinearTransform<Point<float32>>& srcTransform,
    const GpuMatrix<const float32_x2>& map,
    const Point<float32>& mapScaleFactor,
    const Point<float32>& mapValueFactor,
    BorderMode borderMode,
    InterpType interpType,
    const GpuMatrix<DstPixel> dst,
    stdPars(GpuProcessKit)
);
