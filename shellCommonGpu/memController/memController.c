#include "memController.h"

#include "allocation/mallocAllocator/mallocAllocator.h"
#include "allocation/mallocKit.h"
#include "dataAlloc/arrayObjectMemory.inl"
#include "formattedOutput/requireMsg.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "memController/fastAllocator/fastAllocator.h"
#include "numbers/mathIntrinsics.h"
#include "userOutput/printMsg.h"
#include "userOutput/printMsgTrace.h"

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
    stdbool createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdParsNull)
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

    stdbool createTexture(const GpuContext& context, const Point<Space>& size, GpuChannelType chanType, int rank, GpuTextureOwner& result, stdParsNull)
    {
        result.clear();
        require(printMsgTrace(STR("Texture allocation is too slow for temporary memory."), msgErr, stdPassThru));
        returnFalse;
    }

    inline GpuTextureAllocFail(const MsgLogExKit& kit)
        : kit(kit) {}

private:

    MsgLogExKit kit;

};

//================================================================
//
// MemController::~MemController
//
//================================================================

MemController::~MemController()
{
}

//================================================================
//
// MemController::serialize
//
//================================================================

void MemController::serialize(const CfgSerializeKit& kit)
{
    curveCapacity.serialize(kit, STR("Alloc Curve Checker Capacity"));
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
// MemController::dealloc
//
//================================================================

void MemController::dealloc()
{
    curveCapacity = 0;
    cpuCurveBuffer.dealloc();
    gpuCurveBuffer.dealloc();

    stateMemoryIsAllocated = false;
    cpuState.dealloc();
    gpuState.dealloc();

    cpuTemp.dealloc();
    gpuTemp.dealloc();
}

//================================================================
//
// MemController::curveReallocBuffers
//
//================================================================

stdbool MemController::curveReallocBuffers(ReallocActivity& activity, stdPars(ProcessKit))
{
    stdScopedBegin;

    MAKE_MALLOC_ALLOCATOR(kit);
    auto& oldKit = kit;
    auto kit = kitCombine(oldKit, MallocKit(mallocAllocator));

    ////

    if_not (cpuCurveBuffer.maxSize() == curveCapacity)
    {
        ++activity.curveAllocCount;
        require(cpuCurveBuffer.reallocInHeap(curveCapacity, stdPass));
    }

    if_not (gpuCurveBuffer.maxSize() == curveCapacity)
    {
        ++activity.curveAllocCount;
        require(gpuCurveBuffer.reallocInHeap(curveCapacity, stdPass));
    }

    ////

    REQUIRE(cpuCurveBuffer.resize(curveCapacity));
    REQUIRE(gpuCurveBuffer.resize(curveCapacity));

    ////

    stdScopedEnd;
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
        stateUsage.cpuMemSize = cpuState.memory.size();
        stateUsage.gpuMemSize = gpuState.memory.size();
        stateUsage.cpuAlignment = cpuState.alignment;
        stateUsage.gpuAlignment = gpuState.alignment;
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

        bool allocOk = errorBlock(target.realloc(stdPassKitNc(reallocKit)));

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
    // Curve buffers.
    //
    //----------------------------------------------------------------

    require(curveReallocBuffers(stateActivity, stdPass));

    //----------------------------------------------------------------
    //
    // Count state memory / realloc to fake.
    //
    //----------------------------------------------------------------

    using namespace fastAllocator;

    FastAllocator<CpuAddrU, false, true> cpuCounter{cpuCurveBuffer, kit};
    FastAllocator<GpuAddrU, false, true> gpuCounter{gpuCurveBuffer, kit};

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

    auto cpuAlignment = cpuCounter.maxAlign();
    auto gpuAlignment = gpuCounter.maxAlign();

    ////

    auto cpuRecords = cpuCounter.curveSize();
    auto gpuRecords = gpuCounter.curveSize();

    if (curveCapacity)
    {
        if_not (cpuRecords < curveCapacity && gpuRecords < curveCapacity)
        {
            cpuRecords = 0; // Overflow reported, don't check on execution phase.
            gpuRecords = 0;
        }
    }

    REQUIRE(cpuCurveBuffer.resize(cpuRecords));
    REQUIRE(gpuCurveBuffer.resize(gpuRecords));

    //----------------------------------------------------------------
    //
    // Realloc state memory block if required.
    //
    //----------------------------------------------------------------

    bool cpuFastResize = (cpuMemSize <= cpuState.memory.maxSize()) && (cpuAlignment <= cpuState.alignment);
    bool gpuFastResize = (gpuMemSize <= gpuState.memory.maxSize()) && (gpuAlignment <= gpuState.alignment);

    ////

    bool cpuAllocOk = true;

    if_not (cpuFastResize)
    {
        cpuState.memory.dealloc(); // Don't double mem usage.
        cpuState.alignment = 1;

        cpuAllocOk = errorBlock(cpuState.memory.realloc(cpuMemSize, cpuAlignment, alloc.cpuSystemAllocator, stdPassNc));

        if (cpuAllocOk)
            cpuState.alignment = cpuAlignment;
    }

    ////

    bool gpuAllocOk = true;

    if_not (gpuFastResize)
    {
        gpuState.memory.dealloc(); // Don't double mem usage.
        gpuState.alignment = 1;

        gpuAllocOk = errorBlock(gpuState.memory.realloc(gpuMemSize, gpuAlignment, alloc.gpuSystemAllocator, stdPassNc));

        if (gpuAllocOk)
            gpuState.alignment = gpuAlignment;
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

    REQUIRE(cpuState.memory.resize(cpuMemSize));
    FastAllocator<CpuAddrU, true, true> cpuDistributor{cpuState.memory.ptr(), cpuState.memory.size(), cpuCurveBuffer, kit};

    REQUIRE(gpuState.memory.resize(gpuMemSize));
    FastAllocator<GpuAddrU, true, true> gpuDistributor{gpuState.memory.ptr(), gpuState.memory.size(), gpuCurveBuffer, kit};

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
    REQUIRE(cpuDistributor.maxAlign() == cpuAlignment);
    REQUIRE(cpuDistributor.curveSize() == cpuRecords);

    REQUIRE(gpuDistributor.allocatedSpace() == gpuMemSize);
    REQUIRE(gpuDistributor.maxAlign() == gpuAlignment);
    REQUIRE(gpuDistributor.curveSize() == gpuRecords);

    //----------------------------------------------------------------
    //
    // Record successful module state reallocation
    //
    //----------------------------------------------------------------

    stateMemoryIsAllocated = true;

    ////

    stateUsage.cpuMemSize = cpuMemSize;
    stateUsage.gpuMemSize = gpuMemSize;
    stateUsage.cpuAlignment = cpuState.alignment;
    stateUsage.gpuAlignment = gpuState.alignment;

    ////

    returnTrue;
}

//================================================================
//
// MemController::processCountTemp
//
//================================================================

stdbool MemController::processCountTemp(MemControllerProcessTarget& target, MemoryUsage& tempUsage, ReallocActivity& tempActivity, stdPars(ProcessKit))
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
    // Curve buffers.
    //
    //----------------------------------------------------------------

    require(curveReallocBuffers(tempActivity, stdPass));

    //----------------------------------------------------------------
    //
    // Count temp memory / realloc to fake.
    //
    //----------------------------------------------------------------

    using namespace fastAllocator;

    FastAllocator<CpuAddrU, false, false> cpuCounter{cpuCurveBuffer, kit};
    FastAllocator<GpuAddrU, false, false> gpuCounter{gpuCurveBuffer, kit};

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

    REQUIRE(cpuCounter.isValid() && cpuCounter.allocatedSpace() == 0);
    REQUIRE(gpuCounter.isValid() && gpuCounter.allocatedSpace() == 0);

    ////

    auto cpuRecords = cpuCounter.curveSize();
    auto gpuRecords = gpuCounter.curveSize();

    if (curveCapacity)
    {
        if_not (cpuRecords < curveCapacity && gpuRecords < curveCapacity)
        {
            cpuRecords = 0; // Overflow reported, don't check on execution phase.
            gpuRecords = 0;
        }
    }

    REQUIRE(cpuCurveBuffer.resize(cpuRecords));
    REQUIRE(gpuCurveBuffer.resize(gpuRecords));

    ////

    tempUsage.cpuMemSize = cpuCounter.maxAllocatedSpace();
    tempUsage.cpuAlignment = cpuCounter.maxAlign();

    tempUsage.gpuMemSize = gpuCounter.maxAllocatedSpace();
    tempUsage.gpuAlignment = gpuCounter.maxAlign();

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

    bool cpuFastResize = (cpuMemSize <= cpuTemp.memory.maxSize()) && (cpuAlignment <= cpuTemp.alignment);
    bool gpuFastResize = (gpuMemSize <= gpuTemp.memory.maxSize()) && (gpuAlignment <= gpuTemp.alignment);

    ////

    bool cpuAllocOk = true;

    if_not (cpuFastResize)
    {
        cpuTemp.memory.dealloc(); // Don't double mem usage.
        cpuTemp.alignment = 1;

        cpuAllocOk = errorBlock(cpuTemp.memory.realloc(cpuMemSize, cpuAlignment, alloc.cpuSystemAllocator, stdPassNc));

        if (cpuAllocOk)
            cpuTemp.alignment = cpuAlignment;
    }

    ////

    bool gpuAllocOk = true;

    if_not (gpuFastResize)
    {
        gpuTemp.memory.dealloc(); // Don't double mem usage.
        gpuTemp.alignment = 1;

        gpuAllocOk = errorBlock(gpuTemp.memory.realloc(gpuMemSize, gpuAlignment, alloc.gpuSystemAllocator, stdPassNc));

        if (gpuAllocOk)
            gpuTemp.alignment = gpuAlignment;
    }

    ////

    if (cpuAllocOk)
        REQUIRE(cpuTemp.memory.resize(cpuMemSize));

    if (gpuAllocOk)
        REQUIRE(gpuTemp.memory.resize(gpuMemSize));

    //----------------------------------------------------------------
    //
    // On error, report and fail.
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

    FastAllocator<CpuAddrU, true, false> cpuDistributor{cpuTemp.memory.ptr(), cpuTemp.memory.size(), cpuCurveBuffer, kit};
    FastAllocator<GpuAddrU, true, false> gpuDistributor{gpuTemp.memory.ptr(), gpuTemp.memory.size(), gpuCurveBuffer, kit};

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

    REQUIRE(cpuDistributor.isValid());
    REQUIRE(cpuDistributor.allocatedSpace() == 0);
    REQUIRE(cpuDistributor.curveSize() == cpuCurveBuffer.size() || cpuDistributor.curveIsReported());
    tempUsage.cpuMemSize = cpuDistributor.maxAllocatedSpace();
    tempUsage.cpuAlignment = cpuDistributor.maxAlign();

    REQUIRE(gpuDistributor.isValid());
    REQUIRE(gpuDistributor.allocatedSpace() == 0);
    REQUIRE(gpuDistributor.curveSize() == gpuCurveBuffer.size() || gpuDistributor.curveIsReported());
    tempUsage.gpuMemSize = gpuDistributor.maxAllocatedSpace();
    tempUsage.gpuAlignment = gpuDistributor.maxAlign();

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
