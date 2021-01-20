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
stdbool downsampleOneAndThirdConservativeMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

////

template <typename Src, typename Interm, typename Dst>
inline stdbool downsampleOneAndThirdConservative(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit))
    {return downsampleOneAndThirdConservativeMultitask<Src, Interm, Dst>(layeredMatrix(src), layeredMatrix(dst), borderMode, stdPassThru);}

//----------------------------------------------------------------

}
