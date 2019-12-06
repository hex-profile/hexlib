#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"
#include "data/gpuLayeredMatrix.h"

namespace gaussMaskResampling {

//================================================================
//
// downsampleOneAndHalfMaxTasks
//
//================================================================

constexpr int downsampleOneAndHalfMaxTasks = 4;

//================================================================
//
// downsampleOneAndHalfGaussMaskMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>              
stdbool downsampleOneAndHalfGaussMaskMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, bool initial, stdPars(GpuProcessKit));

//================================================================
//
// downsampleOneAndHalfGaussMask
//
//================================================================

template <typename Src, typename Interm, typename Dst>
inline stdbool downsampleOneAndHalfGaussMask(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, bool initial, stdPars(GpuProcessKit))
    {return downsampleOneAndHalfGaussMaskMultitask<Src, Interm, Dst>(layeredMatrix(src), layeredMatrix(dst), borderMode, initial, stdPassThru);}

//----------------------------------------------------------------

}
