#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"
#include "bsplinePrefilterSettings.h"

//================================================================
//
// bsplineCubicPrefilter
//
// Michael Unser prefilter for cubic bspline.
//
// 0.194 ms on GTX780 YUV420 1920x1080
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool bsplineCubicPrefilter
(
    const GpuMatrix<const Src>& src,
    const GpuMatrix<Dst>& dst,
    const Point<float32>& outputFactor,
    BorderMode borderMode,
    stdPars(GpuProcessKit)
);

//================================================================
//
// bsplineCubicUnprefilter
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool bsplineCubicUnprefilter
(
    const GpuMatrix<const Src>& src,
    const GpuMatrix<Dst>& dst,
    const Point<float32>& outputFactor,
    BorderMode borderMode,
    stdPars(GpuProcessKit)
);
