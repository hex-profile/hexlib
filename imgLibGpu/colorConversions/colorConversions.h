#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// convertBgr32ToMono
//
//================================================================

stdbool convertBgr32ToMono
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrix<uint8>& dst,
    stdPars(GpuProcessKit)
);

//================================================================
//
// convertBgr32ToBgr24
//
//================================================================

stdbool convertBgr32ToBgr24
(
    const GpuMatrix<const uint8_x4>& src,
    const GpuMatrix<uint8>& dst,
    stdPars(GpuProcessKit)
);

//================================================================
//
// convertBgr24ToBgr32
//
//================================================================

stdbool convertBgr24ToBgr32Kernel(const GpuMatrix<uint8_x4>& dst, const GpuMatrix<const uint8>& src, stdPars(GpuProcessKit));

inline stdbool convertBgr24ToBgr32(const GpuMatrix<const uint8>& src, const GpuMatrix<uint8_x4>& dst, stdPars(GpuProcessKit))
    {return convertBgr24ToBgr32Kernel(dst, src, stdPassThru);}
