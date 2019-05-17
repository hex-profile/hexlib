#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

namespace conservativeResampling {

//================================================================
//
// upsampleTwice
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool upsampleTwice(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}
