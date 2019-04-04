#pragma once

#include "gpuProcessHeader.h"
#include "numbers/lt/ltBase.h"

//================================================================
//
// convertYuv420ToBgr
//
// 0.22ms FullHD
//
//================================================================

template <typename SrcPixel, typename SrcPixel2, typename DstPixel>
bool convertYuv420ToBgr
(
    const GpuMatrix<const SrcPixel>& srcLuma,
    const GpuMatrix<const SrcPixel2>& srcChroma,
    const Point<Space>& srcOffset,
    const DstPixel& outerColor,
    const GpuMatrix<DstPixel>& dst,
    stdPars(GpuProcessKit)
);
