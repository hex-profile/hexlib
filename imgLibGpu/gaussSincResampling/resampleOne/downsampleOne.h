#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"
#include "data/gpuLayeredMatrix.h"

namespace gaussSincResampling {

//================================================================
//
// downsampleOneConservative
// downsampleOneConservativeMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>              
stdbool downsampleOneConservativeMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

////

template <typename Src, typename Interm, typename Dst>
inline stdbool downsampleOneConservative(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit))
    {return downsampleOneConservativeMultitask<Src, Interm, Dst>(layeredMatrix(src), layeredMatrix(dst), borderMode, stdPassThru);}

//----------------------------------------------------------------

}
