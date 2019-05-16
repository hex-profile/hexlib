#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

//================================================================
//
// upsampleFourTimes
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool upsampleFourTimes
(
    const GpuMatrix<const Src>& src,
    const GpuMatrix<Dst>& dst,
    BorderMode borderMode,
    stdPars(GpuProcessKit)
);

template <typename Src, typename Interm, typename Dst>
stdbool upsampleFourTimesDual
(
    const GpuMatrix<const Src>& srcA,
    const GpuMatrix<Dst>& dstA,
    const GpuMatrix<const Src>& srcB,
    const GpuMatrix<Dst>& dstB,
    BorderMode borderMode,
    stdPars(GpuProcessKit)
);
