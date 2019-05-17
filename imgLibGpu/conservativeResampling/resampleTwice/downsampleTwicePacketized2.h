#pragma once

#include "gpuProcessHeader.h"
#include "imageRead/borderMode.h"

namespace conservativeResampling {

//================================================================
//
// downsampleTwicePacketized2
//
// 0.205 ms FullHD YUV420 on GTX780
//
//================================================================

template <typename Src, typename Interm, typename Dst>
stdbool downsampleTwicePacketized2(const GpuMatrix<const Src>& src, const GpuMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}
