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

bool emuSetSamplerArray
(
    const GpuSamplerLink& sampler,
    GpuAddrU arrayAddr,
    Space arrayByteSize,
    GpuChannelType chanType,
    int rank,
    BorderMode borderMode,
    bool linearInterpolation,
    bool readNormalizedFloat,
    bool normalizedCoords,
    stdPars(ErrorLogKit)
);

//================================================================
//
// emuSetSamplerImage
//
//================================================================

bool emuSetSamplerImage
(
    const GpuSamplerLink& sampler,
    GpuAddrU imageBaseAddr,
    Space imageBytePitch,
    const Point<Space>& imageSize,
    GpuChannelType chanType,
    int rank,
    BorderMode borderMode,
    bool linearInterpolation,
    bool readNormalizedFloat,
    bool normalizedCoords,
    stdPars(ErrorLogKit)
);

//----------------------------------------------------------------

#endif
