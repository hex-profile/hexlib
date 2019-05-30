#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

namespace gaussSincResampling {

//================================================================
//
// upsampleTwiceBalanced
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool upsampleTwiceBalanced(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}
