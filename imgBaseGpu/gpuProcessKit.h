#pragma once

#include "gpuAppliedApi/gpuAppliedKits.h"
#include "cpuFuncKit.h"

//================================================================
//
// GpuProcessKit
//
// Typical kit for a GPU image processing library function.
//
//================================================================

KIT_COMBINE2(GpuProcessKit, CpuFuncKit, GpuAppExecKit);
