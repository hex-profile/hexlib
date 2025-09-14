#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"
#include "data/gpuLayeredMatrix.h"

namespace gaussSincResampling {

//================================================================
//
// downsampleOneAndHalfConservative
//
// (0.20 ms FullHD in packed YUV420 on GTX780)
//
//================================================================

//================================================================
//
// downsampleOneAndHalfConservative
// downsampleOneAndHalfConservativeMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>
void downsampleOneAndHalfConservativeMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

////

template <typename Src, typename Interm, typename Dst>
inline void downsampleOneAndHalfConservative(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit))
    {downsampleOneAndHalfConservativeMultitask<Src, Interm, Dst>(layeredMatrix(src), layeredMatrix(dst), borderMode, stdPassThru);}

//----------------------------------------------------------------

}
