#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

//================================================================
//
// conservativeDownsampleOneAndHalf
//
// (0.20 ms FullHD in packed YUV420 on GTX780)
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool conservativeDownsampleOneAndHalf(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));
