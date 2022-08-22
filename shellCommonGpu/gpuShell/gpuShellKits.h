#pragma once

#include "gpuLayer/gpuLayer.h"

namespace gpuShell {

//================================================================
//
// GpuSystemAllocatorsKit
//
// Regular slow system allocators (for given GPU context)
//
//================================================================

KIT_CREATE3(
    GpuSystemAllocatorsKit,
    AllocatorInterface<CpuAddrU>&, cpuSystemAllocator,
    AllocatorInterface<GpuAddrU>&, gpuSystemAllocator,
    GpuTextureAllocator&, gpuSystemTextureAllocator
);

//================================================================
//
// GpuShellKit
//
//================================================================

using GpuShellKit = KitCombine<GpuInitKit, GpuExecKit, GpuSystemAllocatorsKit, GpuCurrentContextKit, GpuPropertiesKit, GpuCurrentStreamKit>;

//----------------------------------------------------------------

}
