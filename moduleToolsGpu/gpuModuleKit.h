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

KIT_CREATE1(CpuBlockAllocatorKit, BlockAllocatorInterface<CpuAddrU>&, cpuBlockAlloc);
KIT_CREATE1(GpuBlockAllocatorKit, BlockAllocatorInterface<GpuAddrU>&, gpuBlockAlloc);

//================================================================
//
// GpuModuleReallocKit
// GpuModuleProcessKit
//
// Gpu base module interface
//
//================================================================

KIT_COMBINE2(GpuModuleReallocKit, ModuleReallocKit, GpuAppExecKit);
KIT_COMBINE3(GpuModuleProcessKit, ModuleProcessKit, GpuAppExecKit, GpuImageConsoleKit);
