#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

namespace gaussSincResampling {

//================================================================
//
// upsampleThreeTimesBalanced
//
//================================================================

template <typename Src, typename Interm, typename Dst>
void upsampleThreeTimesBalanced
(
    const GpuMatrix<const Src>& src,
    const GpuMatrix<Dst>& dst,
    BorderMode borderMode,
    stdPars(GpuProcessKit)
);

//----------------------------------------------------------------

}
