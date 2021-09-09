#pragma once

#include "gpuAppliedApi/gpuAppliedKits.h"
#include "imageConsole/gpuImageConsoleKit.h"
#include "kits/moduleKit.h"
#include "dataAlloc/memoryAllocator.h"

//================================================================
//
// CpuBlockAllocatorKit
// GpuBlockAllocatorKit
//
//================================================================

KIT_CREATE(CpuBlockAllocatorKit, BlockAllocatorInterface<CpuAddrU>&, cpuBlockAlloc);
KIT_CREATE(GpuBlockAllocatorKit, BlockAllocatorInterface<GpuAddrU>&, gpuBlockAlloc);

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
