#include "memController.h"

#include "memController/fastSpaceAllocator/fastSpaceAllocator.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "userOutput/errorLogEx.h"
#include "userOutput/printMsg.h"
#include "allocation/flatToSpaceAllocatorThunk.h"
#include "numbers/mathIntrinsics.h"

namespace memController {

//================================================================
//
// MODULE_USES_SYSTEM_ALLOCATOR_DIRECTLY
//
// For verification only: Slow!
//
//================================================================

const bool MODULE_USES_SYSTEM_ALLOCATOR_DIRECTLY = false;

//================================================================
//
// GpuTextureAllocIgnore
//
//================================================================

class GpuTextureAllocIgnore : public GpuTextureAllocator
{
    stdbool createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdNullPars)
    {
        result.clear();
        returnTrue;
    }
};

//================================================================
//
// GpuTextureAllocFail
//
//================================================================

class GpuTextureAllocFail : public GpuTextureAllocator
{

public:

    stdbool createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdNullPars)
    {
        result.clear();
        printMsgTrace(kit.errorLogEx, STR("Texture allocation is too slow for temporary memory."), msgErr, stdPassThru);
        returnFalse;
    }

    inline GpuTextureAllocFail(const ErrorLogExKit& kit)
        : kit(kit) {}

private:

    ErrorLogExKit kit;

};

//================================================================
//
// MemController::deinit
//
//================================================================

void MemController::deinit()
{
    stateMemoryIsAllocated = false;

    ////

    cpuStateMemory.dealloc();
    cpuStateAlignment = 0;

    gpuStateMemory.dealloc();
    gpuStateAlignment = 0;

    ////

    cpuTempMemory.dealloc();
    cpuTempAlignment = 0;

    gpuTempMemory.dealloc();
    gpuTempAlignment = 0;
}

//================================================================
//
// memFailReport
//
//================================================================

template <typename AddrU, typename Kit>
inline bool memFailReport(const CharArray& name, AddrU memSize, SpaceU memAlignment, stdPars(Kit))
{
    return printMsg
    (
        kit.localLog, STR("%0: Failed to alloc %1 Mb"),
        name,
        fltf(ldexpv(float32(memSize), -20), 1), memAlignment,
        msgErr
    );
}

//================================================================
//
// MemController::handleStateRealloc
//
// If not successful, dealloc.
//
//================================================================

stdbool MemController::handleStateRealloc(MemControllerReallocTarget& target, const BaseAllocatorsKit& alloc, MemoryUsage& stateUsage, ReallocActivity& stateActivity, stdPars(ProcessKit))
{
    //----------------------------------------------------------------
    //
    // Skip if realloc is not required
    //
    //----------------------------------------------------------------

    if (stateMemoryIsAllocated && target.reallocValid())
    {
        stateUsage.cpuMemSize = cpuStateMemory.size();
        stateUsage.gpuMemSize = gpuStateMemory.size();
        stateUsage.cpuAlignment = cpuStateAlignment;
        stateUsage.gpuAlignment = gpuStateAlignment;
        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Lower the level to deallocated
    //
    //----------------------------------------------------------------

    stateMemoryIsAllocated = false;

    //----------------------------------------------------------------
    //
    // Direct implementation, for testing.
    //
    //----------------------------------------------------------------

    if (MODULE_USES_SYSTEM_ALLOCATOR_DIRECTLY)
    {

        AllocatorState nullState;

        FlatToSpaceAllocatorThunk<CpuAddrU> cpuAllocator(alloc.cpuSystemAllocator, kit);
        FlatToSpaceAllocatorThunk<GpuAddrU> gpuAllocator(alloc.gpuSystemAllocator, kit);

        //
        // Reallocate everything.
        //

        AllocatorObject<CpuAddrU> cpuAllocObject(nullState, cpuAllocator);
        AllocatorObject<GpuAddrU> gpuAllocObject(nullState, gpuAllocator);

        auto reallocKit = kitCombine
        (
            DataProcessingKit(true),
            CpuFastAllocKit(cpuAllocObject),
            CpuBlockAllocatorKit(cpuAllocator),
            GpuFastAllocKit(gpuAllocObject),
            GpuBlockAllocatorKit(gpuAllocator),
            GpuTextureAllocKit(alloc.gpuSystemTextureAllocator)
        );

        bool allocOk = errorBlock(target.realloc(stdPassKit(reallocKit)));

        ////

        if_not (allocOk)
        {
            printMsg(kit.localLog, STR("State memory: System realloc failed"), msgErr);
            returnFalse;
        }

        stateActivity.sysAllocCount++;

        ////

        stateMemoryIsAllocated = true;

        ////

        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Count state memory / realloc to fake (should succeed).
    //
    //----------------------------------------------------------------

    using namespace fastSpaceAllocator;

    AllocatorState cpuCounterState;
    FastAllocatorThunk<CpuAddrU, false, true> cpuCounterInterface(kit);
    cpuCounterInterface.initCountingState(cpuCounterState);

    AllocatorState gpuCounterState;
    FastAllocatorThunk<GpuAddrU, false, true> gpuCounterInterface(kit);
    gpuCounterInterface.initCountingState(gpuCounterState);

    ////

    GpuTextureAllocIgnore gpuTextureCounter;

    ////

    AllocatorObject<CpuAddrU> cpuCounterObject(cpuCounterState, cpuCounterInterface);
    AllocatorObject<GpuAddrU> gpuCounterObject(gpuCounterState, gpuCounterInterface);

    {
        auto reallocKit = kitCombine
        (
            DataProcessingKit(false),
            CpuFastAllocKit(cpuCounterObject),
            CpuBlockAllocatorKit(cpuCounterInterface),
            GpuFastAllocKit(gpuCounterObject),
            GpuBlockAllocatorKit(gpuCounterInterface),
            GpuTextureAllocKit(gpuTextureCounter)
        );

        require(target.realloc(stdPassKit(reallocKit)));
    }

    ////

    CpuAddrU cpuMemSize = cpuCounterInterface.allocatedSpace(cpuCounterState);
    GpuAddrU gpuMemSize = gpuCounterInterface.allocatedSpace(gpuCounterState);

    SpaceU cpuAlignment = cpuCounterInterface.maxAlignment(cpuCounterState);
    SpaceU gpuAlignment = gpuCounterInterface.maxAlignment(gpuCounterState);

    //----------------------------------------------------------------
    //
    // Realloc state memory block if required.
    //
    //----------------------------------------------------------------

    bool cpuFastResize = (cpuMemSize <= cpuStateMemory.maxSize()) && (cpuAlignment <= cpuStateAlignment);
    bool gpuFastResize = (gpuMemSize <= gpuStateMemory.maxSize()) && (gpuAlignment <= gpuStateAlignment);

    ////

    bool cpuAllocOk = true;

    if_not (cpuFastResize)
    {
        cpuStateMemory.dealloc(); // Don't double mem usage.
        cpuStateAlignment = 1;

        cpuAllocOk = errorBlock(cpuStateMemory.realloc(cpuMemSize, cpuAlignment, alloc.cpuSystemAllocator, stdPass));

        if (cpuAllocOk)
            cpuStateAlignment = cpuAlignment;
    }

    ////

    bool gpuAllocOk = true;

    if_not (gpuFastResize)
    {
        gpuStateMemory.dealloc(); // Don't double mem usage.
        gpuStateAlignment = 1;

        gpuAllocOk = errorBlock(gpuStateMemory.realloc(gpuMemSize, gpuAlignment, alloc.gpuSystemAllocator, stdPass));

        if (gpuAllocOk)
            gpuStateAlignment = gpuAlignment;
    }

    //----------------------------------------------------------------
    //
    // Report and fail
    //
    //----------------------------------------------------------------

    if_not (cpuAllocOk) memFailReport(STR("CPU state"), cpuMemSize, cpuAlignment, stdPass);
    if_not (gpuAllocOk) memFailReport(STR("GPU state"), gpuMemSize, gpuAlignment, stdPass);

    if (cpuFastResize && gpuFastResize)
        stateActivity.fastAllocCount++;
    else
        stateActivity.sysAllocCount++;

    require(cpuAllocOk && gpuAllocOk);

    //----------------------------------------------------------------
    //
    // Distibute state memory.
    //
    //----------------------------------------------------------------

    REQUIRE(cpuStateMemory.resize(cpuMemSize));
    REQUIRE(cpuStateMemory.ptr() <= TYPE_MAX(CpuAddrU));
    REQUIRE(cpuStateMemory.size() <= TYPE_MAX(CpuAddrU));

    AllocatorState cpuDistributorState;
    FastAllocatorThunk<CpuAddrU, true, true> cpuDistributorInterface(kit);
    cpuDistributorInterface.initDistribState(cpuDistributorState, CpuAddrU(cpuStateMemory.ptr()), CpuAddrU(cpuStateMemory.size()));

    REQUIRE(gpuStateMemory.resize(gpuMemSize));
    REQUIRE(gpuStateMemory.ptr() <= TYPE_MAX(GpuAddrU));
    REQUIRE(gpuStateMemory.size() <= TYPE_MAX(GpuAddrU));

    AllocatorState gpuDistributorState;
    FastAllocatorThunk<GpuAddrU, true, true> gpuDistributorInterface(kit);
    gpuDistributorInterface.initDistribState(gpuDistributorState, GpuAddrU(gpuStateMemory.ptr()), GpuAddrU(gpuStateMemory.size()));

    ////

    AllocatorObject<CpuAddrU> cpuDistributorObject(cpuDistributorState, cpuDistributorInterface);
    AllocatorObject<GpuAddrU> gpuDistributorObject(gpuDistributorState, gpuDistributorInterface);

    {
        auto reallocKit = kitCombine
        (
            DataProcessingKit(true),
            CpuFastAllocKit(cpuDistributorObject),
            CpuBlockAllocatorKit(cpuDistributorInterface),
            GpuFastAllocKit(gpuDistributorObject),
            GpuBlockAllocatorKit(gpuDistributorInterface),
            GpuTextureAllocKit(alloc.gpuSystemTextureAllocator)
        );

        require(target.realloc(stdPassKit(reallocKit)));
    }

    ////

    REQUIRE(cpuDistributorInterface.allocatedSpace(cpuDistributorState) == cpuMemSize);
    REQUIRE(cpuDistributorInterface.maxAlignment(cpuDistributorState) == cpuAlignment);
    REQUIRE(gpuDistributorInterface.allocatedSpace(gpuDistributorState) == gpuMemSize);
    REQUIRE(gpuDistributorInterface.maxAlignment(gpuDistributorState) == gpuAlignment);

    //----------------------------------------------------------------
    //
    // Record successful module state reallocation
    //
    //----------------------------------------------------------------

    stateMemoryIsAllocated = true;

    ////

    stateUsage.cpuMemSize = cpuMemSize;
    stateUsage.gpuMemSize = gpuMemSize;
    stateUsage.cpuAlignment = cpuStateAlignment;
    stateUsage.gpuAlignment = gpuStateAlignment;

    ////

    returnTrue;
}

//================================================================
//
// MemController::processCountTemp
//
//================================================================

stdbool MemController::processCountTemp(MemControllerProcessTarget& target, MemoryUsage& tempUsage, stdPars(ProcessKit))
{
    //----------------------------------------------------------------
    //
    // Direct-mode allocation for debugging.
    //
    //----------------------------------------------------------------

    if (MODULE_USES_SYSTEM_ALLOCATOR_DIRECTLY)
        returnTrue;

    //----------------------------------------------------------------
    //
    // Count temp memory / realloc to fake (should succeed).
    //
    //----------------------------------------------------------------

    using namespace fastSpaceAllocator;

    AllocatorState cpuCounterState;
    FastAllocatorThunk<CpuAddrU, false, false> cpuCounterInterface(kit);
    cpuCounterInterface.initCountingState(cpuCounterState);

    AllocatorState gpuCounterState;
    FastAllocatorThunk<GpuAddrU, false, false> gpuCounterInterface(kit);
    gpuCounterInterface.initCountingState(gpuCounterState);

    GpuTextureAllocFail gpuTextureCounter(kit);

    ////

    AllocatorObject<CpuAddrU> cpuCounterObject(cpuCounterState, cpuCounterInterface);
    AllocatorObject<GpuAddrU> gpuCounterObject(gpuCounterState, gpuCounterInterface);

    {
        auto processKit = kitCombine
        (
            DataProcessingKit(false),
            CpuFastAllocKit(cpuCounterObject),
            CpuBlockAllocatorKit(cpuCounterInterface),
            GpuFastAllocKit(gpuCounterObject),
            GpuBlockAllocatorKit(gpuCounterInterface),
            GpuTextureAllocKit(gpuTextureCounter)
        );

        require(target.process(stdPassKit(processKit)));
    }

    ////

    REQUIRE(cpuCounterInterface.validState(cpuCounterState) && cpuCounterInterface.allocatedSpace(cpuCounterState) == 0);
    REQUIRE(gpuCounterInterface.validState(gpuCounterState) && gpuCounterInterface.allocatedSpace(gpuCounterState) == 0);

    ////

    tempUsage.cpuMemSize = cpuCounterInterface.maxAllocatedSpace(cpuCounterState);
    tempUsage.cpuAlignment = cpuCounterInterface.maxAlignment(cpuCounterState);

    tempUsage.gpuMemSize = gpuCounterInterface.maxAllocatedSpace(gpuCounterState);
    tempUsage.gpuAlignment = gpuCounterInterface.maxAlignment(gpuCounterState);

    ////

    returnTrue;
}

//================================================================
//
// MemController::handleTempRealloc
//
//================================================================

stdbool MemController::handleTempRealloc(const MemoryUsage& tempUsage, const BaseAllocatorsKit& alloc, ReallocActivity& tempActivity, stdPars(ProcessKit))
{
    if (MODULE_USES_SYSTEM_ALLOCATOR_DIRECTLY)
    {
        tempActivity.sysAllocCount++;
        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Realloc temp system memory block if required.
    //
    //----------------------------------------------------------------

    CpuAddrU cpuMemSize = tempUsage.cpuMemSize;
    GpuAddrU gpuMemSize = tempUsage.gpuMemSize;

    SpaceU cpuAlignment = tempUsage.cpuAlignment;
    SpaceU gpuAlignment = tempUsage.gpuAlignment;

    ////

    bool cpuFastResize = (cpuMemSize <= cpuTempMemory.maxSize()) && (cpuAlignment <= cpuTempAlignment);
    bool gpuFastResize = (gpuMemSize <= gpuTempMemory.maxSize()) && (gpuAlignment <= gpuTempAlignment);

    ////

    bool cpuAllocOk = true;

    if_not (cpuFastResize)
    {
        cpuTempMemory.dealloc(); // Don't double mem usage.
        cpuTempAlignment = 1;

        cpuAllocOk = errorBlock(cpuTempMemory.realloc(cpuMemSize, cpuAlignment, alloc.cpuSystemAllocator, stdPass));

        if (cpuAllocOk)
            cpuTempAlignment = cpuAlignment;
    }

    ////

    bool gpuAllocOk = true;

    if_not (gpuFastResize)
    {
        gpuTempMemory.dealloc(); // Don't double mem usage.
        gpuTempAlignment = 1;

        gpuAllocOk = errorBlock(gpuTempMemory.realloc(gpuMemSize, gpuAlignment, alloc.gpuSystemAllocator, stdPass));

        if (gpuAllocOk)
            gpuTempAlignment = gpuAlignment;
    }

    ////

    if (cpuAllocOk)
        REQUIRE(cpuTempMemory.resize(cpuMemSize));

    if (gpuAllocOk)
        REQUIRE(gpuTempMemory.resize(gpuMemSize));

    //----------------------------------------------------------------
    //
    // On error, report and fail.
    // Prepare distributing allocators.
    //
    //----------------------------------------------------------------

    if_not (cpuAllocOk) memFailReport(STR("CPU temp"), cpuMemSize, cpuAlignment, stdPass);
    if_not (gpuAllocOk) memFailReport(STR("GPU temp"), gpuMemSize, gpuAlignment, stdPass);

    ////

    if (cpuFastResize && gpuFastResize)
        ++tempActivity.fastAllocCount;
    else
        ++tempActivity.sysAllocCount;

    ////

    require(cpuAllocOk && gpuAllocOk);

    ////

    returnTrue;
}

//================================================================
//
// MemController::processAllocTemp
//
//================================================================

stdbool MemController::processAllocTemp(MemControllerProcessTarget& target, const BaseAllocatorsKit& alloc, MemoryUsage& tempUsage, stdPars(ProcessKit))
{
    //----------------------------------------------------------------
    //
    // Direct-mode allocation for debugging.
    //
    //----------------------------------------------------------------

    if (MODULE_USES_SYSTEM_ALLOCATOR_DIRECTLY)
    {
        AllocatorState nullState;

        FlatToSpaceAllocatorThunk<CpuAddrU> cpuAllocator(alloc.cpuSystemAllocator, kit);
        FlatToSpaceAllocatorThunk<GpuAddrU> gpuAllocator(alloc.gpuSystemAllocator, kit);
        GpuTextureAllocFail gpuTextureAllocator(kit);

        AllocatorObject<CpuAddrU> cpuAllocObject(nullState, cpuAllocator);
        AllocatorObject<GpuAddrU> gpuAllocObject(nullState, gpuAllocator);

        ////

        auto processKit = kitCombine
        (
            DataProcessingKit(true),
            CpuFastAllocKit(cpuAllocObject),
            CpuBlockAllocatorKit(cpuAllocator),
            GpuFastAllocKit(gpuAllocObject),
            GpuBlockAllocatorKit(gpuAllocator),
            GpuTextureAllocKit(gpuTextureAllocator)
        );

        require(target.process(stdPassKit(processKit)));

        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Distribute temp memory
    //
    //----------------------------------------------------------------

    using namespace fastSpaceAllocator;

    REQUIRE(cpuTempMemory.ptr() <= TYPE_MAX(CpuAddrU));
    REQUIRE(cpuTempMemory.size() <= TYPE_MAX(CpuAddrU));
    AllocatorState cpuDistributorState;
    FastAllocatorThunk<CpuAddrU, true, false> cpuDistributorInterface(kit);
    cpuDistributorInterface.initDistribState(cpuDistributorState, CpuAddrU(cpuTempMemory.ptr()), CpuAddrU(cpuTempMemory.size()));

    ////

    REQUIRE(gpuTempMemory.ptr() <= TYPE_MAX(GpuAddrU));
    REQUIRE(gpuTempMemory.size() <= TYPE_MAX(GpuAddrU));
    AllocatorState gpuDistributorState;
    FastAllocatorThunk<GpuAddrU, true, false> gpuDistributorInterface(kit);
    gpuDistributorInterface.initDistribState(gpuDistributorState, GpuAddrU(gpuTempMemory.ptr()), GpuAddrU(gpuTempMemory.size()));

    ////

    GpuTextureAllocFail gpuTextureAllocator(kit);

    //----------------------------------------------------------------
    //
    // Process with allocation
    //
    //----------------------------------------------------------------

    AllocatorObject<CpuAddrU> cpuDistributorObject(cpuDistributorState, cpuDistributorInterface);
    AllocatorObject<GpuAddrU> gpuDistributorObject(gpuDistributorState, gpuDistributorInterface);

    auto processKit = kitCombine
    (
        DataProcessingKit(true),
        CpuFastAllocKit(cpuDistributorObject),
        CpuBlockAllocatorKit(cpuDistributorInterface),
        GpuFastAllocKit(gpuDistributorObject),
        GpuBlockAllocatorKit(gpuDistributorInterface),
        GpuTextureAllocKit(gpuTextureAllocator)
    );

    require(target.process(stdPassKit(processKit)));

    //----------------------------------------------------------------
    //
    // Check
    //
    //----------------------------------------------------------------

    REQUIRE(cpuDistributorInterface.validState(cpuDistributorState));
    REQUIRE(cpuDistributorInterface.allocatedSpace(cpuDistributorState) == 0);
    tempUsage.cpuMemSize = cpuDistributorInterface.maxAllocatedSpace(cpuDistributorState);
    tempUsage.cpuAlignment = cpuDistributorInterface.maxAlignment(cpuDistributorState);

    REQUIRE(gpuDistributorInterface.validState(gpuDistributorState));
    REQUIRE(gpuDistributorInterface.allocatedSpace(gpuDistributorState) == 0);
    tempUsage.gpuMemSize = gpuDistributorInterface.maxAllocatedSpace(gpuDistributorState);
    tempUsage.gpuAlignment = gpuDistributorInterface.maxAlignment(gpuDistributorState);

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
