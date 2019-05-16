#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// downsampleTwiceCubic
//
// 0.115 ms FullHD
//
//================================================================

template <typename Src, typename Dst>
stdbool downsampleTwiceCubic(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, const Point<Space>& srcOfs, stdPars(GpuProcessKit));
