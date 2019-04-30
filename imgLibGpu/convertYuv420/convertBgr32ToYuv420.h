#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// convertBgr32ToYuv420
//
// Supports both vector chroma and planar chroma output:
// any destination matrix may be empty.
//
// 0.313ms FullHD on GTX 780
//
//================================================================

template <typename DstPixel, typename DstPixel2>
stdbool convertBgr32ToYuv420
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrix<DstPixel>& dstLuma,
    const GpuMatrix<DstPixel2>& dstChroma,
    const GpuMatrix<DstPixel>& dstChromaU,
    const GpuMatrix<DstPixel>& dstChromaV,
    stdPars(GpuProcessKit)
);
