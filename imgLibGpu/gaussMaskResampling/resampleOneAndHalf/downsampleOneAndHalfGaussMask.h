#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

namespace gaussMaskResampling {

//================================================================
//
// downsampleOneAndHalfGaussMaskInitial
// downsampleOneAndHalfGaussMaskNextSigma
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool downsampleOneAndHalfGaussMaskInitial(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

template <typename Src, typename Interm, typename Dst>
stdbool downsampleOneAndHalfGaussMaskSustaining(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}
