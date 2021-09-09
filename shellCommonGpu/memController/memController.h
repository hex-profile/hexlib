#pragma once

#include "allocation/flatMemoryAllocator.h"
#include "gpuModuleHeader.h"
#include "memController/flatMemoryHolder.h"
#include "numbers/interface/numberInterface.h"

namespace memController {

//================================================================
//
// FastAllocToolkit
//
//================================================================

using FastAllocToolkit = KitCombine<DataProcessingKit, CpuFastAllocKit, CpuBlockAllocatorKit, GpuFastAllocKit, GpuBlockAllocatorKit>;

//================================================================
//
// MemoryUsage
//
//================================================================

struct MemoryUsage
{
    CpuAddrU cpuMemSize = 0;
    SpaceU cpuAlignment = 1;

    GpuAddrU gpuMemSize = 0;
    SpaceU gpuAlignment = 1;
};

//----------------------------------------------------------------

sysinline MemoryUsage maxOf(const MemoryUsage& A, const MemoryUsage& B)
{
    MemoryUsage tmp;

    tmp.cpuMemSize = maxv(A.cpuMemSize, B.cpuMemSize);
    tmp.gpuMemSize = maxv(A.gpuMemSize, B.gpuMemSize);
    tmp.cpuAlignment = maxv(A.cpuAlignment, B.cpuAlignment);
    tmp.gpuAlignment = maxv(A.gpuAlignment, B.gpuAlignment);

    return tmp;
}

//----------------------------------------------------------------

sysinline bool operator ==(const MemoryUsage& A, const MemoryUsage& B)
{
    return
        A.cpuMemSize == B.cpuMemSize &&
        A.cpuAlignment == B.cpuAlignment &&
        A.gpuMemSize == B.gpuMemSize &&
        A.gpuAlignment == B.gpuAlignment;
}

//================================================================
//
// ReallocActivity
//
//================================================================

struct ReallocActivity
{
    Space fastAllocCount = 0;
    Space sysAllocCount = 0;
};

//================================================================
//
// MemControllerReallocTarget
//
//================================================================

struct MemControllerReallocTarget
{
    virtual bool reallocValid() const =0;
    virtual stdbool realloc(stdPars(FastAllocToolkit)) =0;
};

//================================================================
//
// MemControllerProcessTarget
//
//================================================================

struct MemControllerProcessTarget
{
    virtual stdbool process(stdPars(FastAllocToolkit)) =0;
};

//================================================================
//
// MemController
//
// Implements:
// * Fast memory reallocations for CPU/GPU memory.
//
// Uses:
// * Target object is given via abstract interface.
// * Basic slow CPU/GPU alloc.
// * Error reporting, user message interface.
//
//================================================================

KIT_CREATE3(
    BaseAllocatorsKit,
    FlatMemoryAllocator<CpuAddrU>&, cpuSystemAllocator,
    FlatMemoryAllocator<GpuAddrU>&, gpuSystemAllocator,
    GpuTextureAllocator&, gpuSystemTextureAllocator
);

//================================================================
//
// ProcessKit
//
//================================================================

using ProcessKit = KitCombine<ErrorLogKit, ErrorLogExKit, ProfilerKit, MsgLogsKit>;

//================================================================
//
// MemController
//
//================================================================

class MemController
{

public:

    void deinit();

public:

    //
    // If the module state memory reallocation is required,
    // counts required state memory, reallocates memory pools if necessary,
    // and reallocates the module state with real memory distribution.
    //

    stdbool handleStateRealloc(MemControllerReallocTarget& target, const BaseAllocatorsKit& alloc, MemoryUsage& stateUsage, ReallocActivity& stateActivity, stdPars(ProcessKit));

    //
    // Counts temporary memory required for processing.
    //

    stdbool processCountTemp(MemControllerProcessTarget& target, MemoryUsage& tempUsage, stdPars(ProcessKit));

    //
    // Given the required temp memory size, reallocates memory pools if necessary.
    //

    stdbool handleTempRealloc(const MemoryUsage& tempUsage, const BaseAllocatorsKit& alloc, ReallocActivity& tempActivity, stdPars(ProcessKit));

    //
    // Calls processing with real memory distribution.
    //

    stdbool processAllocTemp(MemControllerProcessTarget& target, const BaseAllocatorsKit& alloc, MemoryUsage& tempUsage, stdPars(ProcessKit));

private:

    //
    // Module state memory
    //

    bool stateMemoryIsAllocated = false;

    FlatMemoryHolder<CpuAddrU> cpuStateMemory;
    SpaceU cpuStateAlignment = 1;

    FlatMemoryHolder<GpuAddrU> gpuStateMemory;
    SpaceU gpuStateAlignment = 1;

    //
    // Module temp memory
    //

    FlatMemoryHolder<CpuAddrU> cpuTempMemory;
    SpaceU cpuTempAlignment = 1;

    FlatMemoryHolder<GpuAddrU> gpuTempMemory;
    SpaceU gpuTempAlignment = 1;

};

//----------------------------------------------------------------

}

using memController::MemController;
using memController::MemoryUsage;
using memController::ReallocActivity;
using memController::MemControllerProcessTarget;
using memController::MemControllerReallocTarget;

