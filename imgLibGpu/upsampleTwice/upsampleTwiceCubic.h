#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// upsampleTwiceCubic
//
//================================================================

template <typename Src, typename Dst>
stdbool upsampleTwiceCubic(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, stdPars(GpuProcessKit));
