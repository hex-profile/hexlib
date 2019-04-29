#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"
#include "imageRead/interpType.h"
#include "numbers/lt/ltBase.h"

//================================================================
//
// warpImage
//
// Image interpolation and border mode are specified.
// Map parameters are always INTERP_LINEAR and BORDER_CLAMP.
//
//================================================================

template <typename Pixel>
bool warpImage
(
    const GpuMatrix<const Pixel>& src,
    const LinearTransform<Point<float32>>& srcTransform,
    const GpuMatrix<const float32_x2>& map,
    const Point<float32>& mapScaleFactor,
    const Point<float32>& mapValueFactor,
    BorderMode borderMode,
    InterpType interpType,
    const GpuMatrix<Pixel> dst,
    stdPars(GpuProcessKit)
);
