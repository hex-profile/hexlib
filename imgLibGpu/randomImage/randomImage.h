#pragma once

#include "gpuProcessHeader.h"
#include "rndgen/rndgenBase.h"

//================================================================
//
// initializeRandomStateMatrix
//
//================================================================

void initializeRandomStateMatrix(const GpuMatrix<RndgenState>& state, const uint32& frameIndex, const uint32& xorValue, stdPars(GpuProcessKit));
