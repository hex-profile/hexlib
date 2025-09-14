#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

namespace gaussSincResampling {

//================================================================
//
// upsampleOneAndQuarterBalanced
//
//================================================================

template <typename Src, typename Interm, typename Dst>
void upsampleOneAndQuarterBalanced(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}
