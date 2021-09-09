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

using GpuProcessKit = KitCombine<CpuFuncKit, GpuAppExecKit>;
