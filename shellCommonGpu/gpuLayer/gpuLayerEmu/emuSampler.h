#pragma once

#if HEXLIB_PLATFORM == 0

#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"
#include "gpuAppliedApi/gpuSamplerSetup.h"

//================================================================
//
// emuSetSamplerArray
//
//================================================================

stdbool emuSetSamplerArray
(
    const GpuSamplerLink& sampler,
    GpuAddrU arrayAddr,
    Space arrayByteSize,
    GpuChannelType chanType,
    int rank,
    BorderMode borderMode,
    LinearInterpolation linearInterpolation,
    ReadNormalizedFloat readNormalizedFloat,
    NormalizedCoords normalizedCoords,
    stdPars(ErrorLogKit)
);

//================================================================
//
// emuSetSamplerImage
//
//================================================================

stdbool emuSetSamplerImage
(
    const GpuSamplerLink& sampler,
    GpuAddrU imageBaseAddr,
    Space imageBytePitch,
    const Point<Space>& imageSize,
    GpuChannelType chanType,
    int rank,
    BorderMode borderMode,
    LinearInterpolation linearInterpolation,
    ReadNormalizedFloat readNormalizedFloat,
    NormalizedCoords normalizedCoords,
    stdPars(ErrorLogKit)
);

//----------------------------------------------------------------

#endif
