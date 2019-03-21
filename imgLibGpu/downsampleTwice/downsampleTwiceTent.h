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
bool downsampleTwiceTent(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, stdPars(GpuProcessKit));
