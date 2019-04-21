#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// downsampleTwiceTent
//
// 0.095 ms FullHD
//
//================================================================

template <typename Src, typename Dst>
stdbool downsampleTwiceTent(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, stdPars(GpuProcessKit));
