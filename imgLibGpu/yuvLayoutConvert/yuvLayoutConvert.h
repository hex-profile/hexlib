#pragma once

#include "gpuProcessHeader.h"
#include "data/gpuImageYuv.h"
#include "yuvLayoutConvert/yuvLayoutConvertCommon.h"

namespace yuvLayoutConvert {

//================================================================
//
// convertRawToYuv420
// convertYuv420ToRaw
//
//================================================================

template <typename RawPixel>
stdbool convertRawToYuv420(const GpuArray<const RawPixel>& src, const GpuImageYuv<Luma>& dst, stdPars(GpuProcessKit));

template <typename RawPixel>
stdbool convertYuv420ToRaw(const GpuImageYuv<const Luma>& src, const GpuArray<RawPixel>& dst, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}
