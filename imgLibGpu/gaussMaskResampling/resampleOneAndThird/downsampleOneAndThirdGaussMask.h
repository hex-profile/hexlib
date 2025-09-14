#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"
#include "data/gpuLayeredMatrix.h"

namespace gaussMaskResampling {

//================================================================
//
// downsampleOneAndThirdGaussMaskMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>
void downsampleOneAndThirdGaussMaskMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, bool initial, stdPars(GpuProcessKit));

//================================================================
//
// downsampleOneAndThirdGaussMask
//
//================================================================

template <typename Src, typename Interm, typename Dst>
inline void downsampleOneAndThirdGaussMask(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, bool initial, stdPars(GpuProcessKit))
    {downsampleOneAndThirdGaussMaskMultitask<Src, Interm, Dst>(layeredMatrix(src), layeredMatrix(dst), borderMode, initial, stdPassThru);}

//----------------------------------------------------------------

}
