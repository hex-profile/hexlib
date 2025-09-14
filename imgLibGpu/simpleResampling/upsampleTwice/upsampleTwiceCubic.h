#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// upsampleTwiceCubic
//
//================================================================

template <typename Src, typename Dst>
void upsampleTwiceCubic(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, stdPars(GpuProcessKit));
