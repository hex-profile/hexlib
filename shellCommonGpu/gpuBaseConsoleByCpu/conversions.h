#pragma once

#include "gpuProcessHeader.h"

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
