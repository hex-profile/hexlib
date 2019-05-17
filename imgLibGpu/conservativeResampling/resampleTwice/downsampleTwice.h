#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

namespace conservativeResampling {

//================================================================
//
// downsampleTwice
//
// 0.30 ms FullHD in YUV420 on GTX780
//
// For big images, the packetized version is faster,
// but for small images this version is more parallel,
// as each pixel is processed by independent thread.
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool downsampleTwice(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}
