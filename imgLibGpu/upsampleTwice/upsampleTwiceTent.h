#pragma once

#include "gpuProcessHeader.h"

//================================================================
//
// upsampleTwiceTent
//
// Gauss-like pyramidal upsampling, using 2 source taps.
//
//================================================================

bool upsampleTwiceTent(const GpuMatrix<const uint8>& src, const GpuMatrix<uint8>& dst, stdPars(GpuProcessKit));
