#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

namespace gaussSincResampling {

//================================================================
//
// downsampleOneAndHalfConservative
//
// (0.20 ms FullHD in packed YUV420 on GTX780)
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool downsampleOneAndHalfConservative
(
    const GpuMatrix<const Src>& src,
    const GpuMatrix<Dst>& dst,
    BorderMode borderMode,
    stdPars(GpuProcessKit)
);

template <typename Src, typename Interm, typename Dst>
stdbool downsampleOneAndHalfConservative2
(
    const GpuMatrix<const Src>& src0, const GpuMatrix<Dst>& dst0,
    const GpuMatrix<const Src>& src1, const GpuMatrix<Dst>& dst1,
    BorderMode borderMode,
    stdPars(GpuProcessKit)
);

template <typename Src, typename Interm, typename Dst>
stdbool downsampleOneAndHalfConservative3
(
    const GpuMatrix<const Src>& src0, const GpuMatrix<Dst>& dst0,
    const GpuMatrix<const Src>& src1, const GpuMatrix<Dst>& dst1,
    const GpuMatrix<const Src>& src2, const GpuMatrix<Dst>& dst2,
    BorderMode borderMode, 
    stdPars(GpuProcessKit)
);

template <typename Src, typename Interm, typename Dst>
stdbool downsampleOneAndHalfConservative4
(
    const GpuMatrix<const Src>& src0, const GpuMatrix<Dst>& dst0,
    const GpuMatrix<const Src>& src1, const GpuMatrix<Dst>& dst1,
    const GpuMatrix<const Src>& src2, const GpuMatrix<Dst>& dst2,
    const GpuMatrix<const Src>& src3, const GpuMatrix<Dst>& dst3,
    BorderMode borderMode, 
    stdPars(GpuProcessKit)
);

//================================================================
//
// downsampleOneAndHalfBalanced
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool downsampleOneAndHalfBalanced(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}
