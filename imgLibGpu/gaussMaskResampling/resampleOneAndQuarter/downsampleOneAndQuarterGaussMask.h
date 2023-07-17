#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"
#include "data/gpuLayeredMatrix.h"

namespace gaussMaskResampling {

//================================================================
//
// downsampleOneAndQuarterGaussMaskMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool downsampleOneAndQuarterGaussMaskMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, bool initial, stdPars(GpuProcessKit));

//================================================================
//
// downsampleOneAndQuarterGaussMask
//
//================================================================

template <typename Src, typename Interm, typename Dst>
inline stdbool downsampleOneAndQuarterGaussMask(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, bool initial, stdPars(GpuProcessKit))
    {return downsampleOneAndQuarterGaussMaskMultitask<Src, Interm, Dst>(layeredMatrix(src), layeredMatrix(dst), borderMode, initial, stdPassThru);}

//----------------------------------------------------------------

}
