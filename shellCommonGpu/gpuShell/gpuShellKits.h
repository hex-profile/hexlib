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
    FlatMemoryAllocator<CpuAddrU>&, cpuSystemAllocator,
    FlatMemoryAllocator<GpuAddrU>&, gpuSystemAllocator,
    GpuTextureAllocator&, gpuSystemTextureAllocator
);

//================================================================
//
// GpuShellKit
//
//================================================================

KIT_COMBINE6(GpuShellKit, GpuInitKit, GpuExecKit, GpuSystemAllocatorsKit, GpuCurrentContextKit, GpuPropertiesKit, GpuCurrentStreamKit);

//----------------------------------------------------------------

}
