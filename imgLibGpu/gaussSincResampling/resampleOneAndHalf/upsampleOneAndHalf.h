#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

namespace gaussSincResampling {

//================================================================
//
// upsampleOneAndHalfConservative
//
// 0.143 ms 1280x720 -> 1920x1080 Monochrome on GTX780
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool upsampleOneAndHalfConservative(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

//================================================================
//
// upsampleOneAndHalfBalanced
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool upsampleOneAndHalfBalanced(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}
