#pragma once

#include "gpuAppliedApi/gpuAppliedKits.h"
#include "imageConsole/gpuImageConsoleKit.h"
#include "kits/moduleKit.h"
#include "dataAlloc/memoryAllocator.h"

//================================================================
//
// GpuModuleReallocKit
// GpuModuleProcessKit
//
// Gpu base module interface
//
//================================================================

using GpuModuleReallocKit = KitCombine<ModuleReallocKit, GpuAppExecKit>;
using GpuModuleProcessKit = KitCombine<ModuleProcessKit, GpuAppExecKit, GpuImageConsoleKit>;
