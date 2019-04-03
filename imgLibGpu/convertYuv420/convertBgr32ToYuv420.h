#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// convertBgr32ToYuv420
//
// 0.313ms FullHd on GTX 780
//
//================================================================

template <typename DstPixel, typename DstPixel2>
bool convertBgr32ToYuv420
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrix<DstPixel>& dstLuma,
    const GpuMatrix<DstPixel2>& dstChroma,
    stdPars(GpuProcessKit)
);
