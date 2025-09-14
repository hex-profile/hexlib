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
    const GpuMatrixAP<const uint8_x4>& src,
    const GpuMatrixAP<DstPixel>& dstLuma,
    const GpuMatrixAP<DstPixel2>& dstChroma,
    const GpuMatrixAP<DstPixel>& dstChromaU,
    const GpuMatrixAP<DstPixel>& dstChromaV,
    stdPars(GpuProcessKit)
);
