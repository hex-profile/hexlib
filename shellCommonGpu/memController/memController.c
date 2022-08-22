#include "memController.h"

#include "memController/fastAllocator/fastAllocator.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "userOutput/errorLogEx.h"
#include "userOutput/printMsg.h"
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
inline bool memFailReport(const CharArray& name, AddrU memSize, AddrU memAlignment, stdPars(Kit))
{
    return printMsg
    (
        kit.localLog, STR("%0: Failed to alloc %1 Mb"),
        name,
        fltf(ldexpv(float32(memSize), -20), 1),
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
        auto& cpuAllocator = alloc.cpuSystemAllocator;
        auto& gpuAllocator = alloc.gpuSystemAllocator;

        //
        // Reallocate everything.
        //

        auto reallocKit = kitCombine
        (
            DataProcessingKit(true),
            CpuFastAllocKit(cpuAllocator),
            GpuFastAllocKit(gpuAllocator),
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

    using namespace fastAllocator;

    FastAllocator<CpuAddrU, false, true> cpuCounter{kit};
    FastAllocator<GpuAddrU, false, true> gpuCounter{kit};

    ////

    GpuTextureAllocIgnore gpuTextureCounter;

    ////

    {
        auto reallocKit = kitCombine
        (
            DataProcessingKit(false),
            CpuFastAllocKit(cpuCounter),
            GpuFastAllocKit(gpuCounter),
            GpuTextureAllocKit(gpuTextureCounter)
        );

        require(target.realloc(stdPassKit(reallocKit)));
    }

    ////

    auto cpuMemSize = cpuCounter.allocatedSpace();
    auto gpuMemSize = gpuCounter.allocatedSpace();

    auto cpuAlignment = cpuCounter.maxAlignment();
    auto gpuAlignment = gpuCounter.maxAlignment();

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
    FastAllocator<CpuAddrU, true, true> cpuDistributor{cpuStateMemory.ptr(), cpuStateMemory.size(), kit};

    REQUIRE(gpuStateMemory.resize(gpuMemSize));
    FastAllocator<GpuAddrU, true, true> gpuDistributor{gpuStateMemory.ptr(), gpuStateMemory.size(), kit};

    ////

    {
        auto reallocKit = kitCombine
        (
            DataProcessingKit(true),
            CpuFastAllocKit(cpuDistributor),
            GpuFastAllocKit(gpuDistributor),
            GpuTextureAllocKit(alloc.gpuSystemTextureAllocator)
        );

        require(target.realloc(stdPassKit(reallocKit)));
    }

    ////

    REQUIRE(cpuDistributor.allocatedSpace() == cpuMemSize);
    REQUIRE(cpuDistributor.maxAlignment() == cpuAlignment);
    REQUIRE(gpuDistributor.allocatedSpace() == gpuMemSize);
    REQUIRE(gpuDistributor.maxAlignment() == gpuAlignment);

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

    using namespace fastAllocator;

    FastAllocator<CpuAddrU, false, false> cpuCounter{kit};
    FastAllocator<GpuAddrU, false, false> gpuCounter{kit};

    GpuTextureAllocFail gpuTextureCounter(kit);

    ////

    auto processKit = kitCombine
    (
        DataProcessingKit(false),
        CpuFastAllocKit(cpuCounter),
        GpuFastAllocKit(gpuCounter),
        GpuTextureAllocKit(gpuTextureCounter)
    );

    require(target.process(stdPassKit(processKit)));

    ////

    REQUIRE(cpuCounter.validState() && cpuCounter.allocatedSpace() == 0);
    REQUIRE(gpuCounter.validState() && gpuCounter.allocatedSpace() == 0);

    ////

    tempUsage.cpuMemSize = cpuCounter.maxAllocatedSpace();
    tempUsage.cpuAlignment = cpuCounter.maxAlignment();

    tempUsage.gpuMemSize = gpuCounter.maxAllocatedSpace();
    tempUsage.gpuAlignment = gpuCounter.maxAlignment();

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

    auto cpuMemSize = tempUsage.cpuMemSize;
    auto gpuMemSize = tempUsage.gpuMemSize;

    auto cpuAlignment = tempUsage.cpuAlignment;
    auto gpuAlignment = tempUsage.gpuAlignment;

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
        auto& cpuAllocator = alloc.cpuSystemAllocator;
        auto& gpuAllocator = alloc.gpuSystemAllocator;
        GpuTextureAllocFail gpuTextureAllocator(kit);

        ////

        auto processKit = kitCombine
        (
            DataProcessingKit(true),
            CpuFastAllocKit(cpuAllocator),
            GpuFastAllocKit(gpuAllocator),
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

    using namespace fastAllocator;

    FastAllocator<CpuAddrU, true, false> cpuDistributor{cpuTempMemory.ptr(), cpuTempMemory.size(), kit};
    FastAllocator<GpuAddrU, true, false> gpuDistributor{gpuTempMemory.ptr(), gpuTempMemory.size(), kit};

    GpuTextureAllocFail gpuTextureAllocator(kit);

    //----------------------------------------------------------------
    //
    // Process with allocation
    //
    //----------------------------------------------------------------

    auto processKit = kitCombine
    (
        DataProcessingKit(true),
        CpuFastAllocKit(cpuDistributor),
        GpuFastAllocKit(gpuDistributor),
        GpuTextureAllocKit(gpuTextureAllocator)
    );

    require(target.process(stdPassKit(processKit)));

    //----------------------------------------------------------------
    //
    // Check
    //
    //----------------------------------------------------------------

    REQUIRE(cpuDistributor.validState());
    REQUIRE(cpuDistributor.allocatedSpace() == 0);
    tempUsage.cpuMemSize = cpuDistributor.maxAllocatedSpace();
    tempUsage.cpuAlignment = cpuDistributor.maxAlignment();

    REQUIRE(gpuDistributor.validState());
    REQUIRE(gpuDistributor.allocatedSpace() == 0);
    tempUsage.gpuMemSize = gpuDistributor.maxAllocatedSpace();
    tempUsage.gpuAlignment = gpuDistributor.maxAlignment();

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
