#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

namespace gaussSincResampling {

//================================================================
//
// downsampleOneAndHalfConservative
//
// (0.20 ms FullHD in packed YUV420 on GTX780)
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool downsampleOneAndHalfConservative(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

//================================================================
//
// downsampleOneAndHalfBalanced
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool downsampleOneAndHalfBalanced(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}
