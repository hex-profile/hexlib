#pragma once

#include "gpuProcessHeader.h"
#include "types/lt/ltBase.h"

//================================================================
//
// convertYuv420ToBgr
//
// 0.22ms FullHD
//
//================================================================

template <typename SrcPixel, typename SrcPixel2, typename DstPixel>
void convertYuv420ToBgr
(
    const GpuMatrix<const SrcPixel>& srcLuma,
    const GpuMatrix<const SrcPixel2>& srcChromaPacked,
    const GpuMatrix<const SrcPixel>& srcChromaU,
    const GpuMatrix<const SrcPixel>& srcChromaV,
    const Point<Space>& srcOffset,
    const DstPixel& outerColor,
    const GpuMatrixAP<DstPixel>& dst,
    stdPars(GpuProcessKit)
);
