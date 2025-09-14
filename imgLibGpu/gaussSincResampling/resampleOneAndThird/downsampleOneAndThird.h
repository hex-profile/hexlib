#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"
#include "data/gpuLayeredMatrix.h"

namespace gaussSincResampling {

//================================================================
//
// downsampleOneAndThirdConservative
// downsampleOneAndThirdConservativeMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>
void downsampleOneAndThirdConservativeMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

////

template <typename Src, typename Interm, typename Dst>
inline void downsampleOneAndThirdConservative(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit))
    {downsampleOneAndThirdConservativeMultitask<Src, Interm, Dst>(layeredMatrix(src), layeredMatrix(dst), borderMode, stdPassThru);}

//----------------------------------------------------------------

}
