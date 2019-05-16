#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

//================================================================
//
// upsampleOneAndHalf
//
// 0.143 ms 1280x720 -> 1920x1080 Monochrome on GTX780
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool upsampleOneAndHalf(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));
