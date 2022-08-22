#pragma once

#include "gpuModuleHeader.h"
#include "memController/flatMemoryHolder.h"
#include "numbers/interface/numberInterface.h"
#include "cfgTools/numericVar.h"
#include "dataAlloc/arrayObjectMemory.h"

namespace memController {

//================================================================
//
// FastAllocToolkit
//
//================================================================

using FastAllocToolkit = KitCombine<DataProcessingKit, CpuFastAllocKit, GpuFastAllocKit>;

//================================================================
//
// MemoryUsage
//
//================================================================

struct MemoryUsage
{
    CpuAddrU cpuMemSize = 0;
    CpuAddrU cpuAlignment = 1;

    GpuAddrU gpuMemSize = 0;
    GpuAddrU gpuAlignment = 1;
};

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
    Space curveAllocCount = 0;
};

//================================================================
//
// uncommonActivity
//
//================================================================

inline bool uncommonActivity(const ReallocActivity& stateActivity, const ReallocActivity& tempActivity)
{
    return 
        stateActivity.sysAllocCount ||
        stateActivity.fastAllocCount || // Report even fast realloc of the state.
        stateActivity.curveAllocCount ||
        tempActivity.sysAllocCount ||
        tempActivity.curveAllocCount;
}

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
    AllocatorInterface<CpuAddrU>&, cpuSystemAllocator,
    AllocatorInterface<GpuAddrU>&, gpuSystemAllocator,
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
// MemState
//
//================================================================

template <typename AddrU>
struct MemState
{
    FlatMemoryHolder<AddrU> memory;
    AddrU alignment = 1;
};

//================================================================
//
// MemController
//
//================================================================

class MemController
{

public:

    ~MemController();

public:

    void serialize(const CfgSerializeKit& kit);

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

    stdbool processCountTemp(MemControllerProcessTarget& target, MemoryUsage& tempUsage, ReallocActivity& tempActivity, stdPars(ProcessKit));

    //
    // Given the required temp memory size, reallocates memory pools if necessary.
    //

    stdbool handleTempRealloc(const MemoryUsage& tempUsage, const BaseAllocatorsKit& alloc, ReallocActivity& tempActivity, stdPars(ProcessKit));

    //
    // Calls processing with real memory distribution.
    //

    stdbool processAllocTemp(MemControllerProcessTarget& target, const BaseAllocatorsKit& alloc, MemoryUsage& tempUsage, stdPars(ProcessKit));

private:

    stdbool curveReallocBuffers(ReallocActivity& activity, stdPars(ProcessKit));

private:

    NumericVar<Space> curveCapacity{0, spaceMax, 0};
    ArrayObjectMemory<CpuAddrU> cpuCurveBuffer;
    ArrayObjectMemory<GpuAddrU> gpuCurveBuffer;

    //
    // Module state memory.
    //

    bool stateMemoryIsAllocated = false;

    MemState<CpuAddrU> cpuState;
    MemState<GpuAddrU> gpuState;

    //
    // Module temp memory.
    //

    MemState<CpuAddrU> cpuTemp;
    MemState<GpuAddrU> gpuTemp;

};

//----------------------------------------------------------------

}

using memController::MemController;
using memController::MemoryUsage;
using memController::ReallocActivity;
using memController::MemControllerProcessTarget;
using memController::MemControllerReallocTarget;

