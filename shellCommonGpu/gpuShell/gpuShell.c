#include "gpuShell.h"

#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"
#include "userOutput/paramMsg.h"
#include "errorLog/debugBreak.h"

namespace gpuShell {

//================================================================
//
// GpuContextHelper::serialize
//
//================================================================

bool GpuContextHelper::serialize(const CfgSerializeKit& kit)
{
    bool indexSteady = gpuDeviceIndex.serialize(kit, STR("GPU Device Index"));

    bool schedulingSteady = gpuScheduling.serialize
    (
        kit, STR("GPU Scheduling"),

        {STR("Spin"), STR("")},
        {STR("Yield"), STR("")},
        {STR("Block"), STR("")},
        {STR("Auto"), STR("")},

        false
    );

    return schedulingSteady && schedulingSteady;
}

//================================================================
//
// GpuContextHelper::createContext
//
//================================================================

void GpuContextHelper::createContext(GpuProperties& gpuProperties, GpuContextOwner& gpuContext, stdPars(InitKit))
{
    int32 gpuDeviceCount = 0;
    kit.gpuInitialization.getDeviceCount(gpuDeviceCount, stdPass);

    REQUIRE_EX(gpuDeviceCount >= 1, printMsg(kit.msgLog, STR("No GPU devices found"), msgErr));

    //
    // Use selected GPU device
    //

    REQUIRE_EX(gpuDeviceIndex >= 0 && gpuDeviceIndex < gpuDeviceCount,
        printMsg(kit.msgLog, STR("GPU device index % cannot be selected (totally % GPUs found)"),
            gpuDeviceIndex(), gpuDeviceCount, msgErr));

    kit.gpuInitialization.getProperties(gpuDeviceIndex, gpuProperties, stdPass);
    REMEMBER_CLEANUP_EX(gpuPropertiesCleanup, gpuProperties = GpuProperties{});

    //
    // Context
    //

    void* baseContext = 0;
    kit.gpuContextCreation.createContext(gpuDeviceIndex, gpuScheduling(), gpuContext, baseContext, stdPass);
    REMEMBER_CLEANUP_EX(gpuContextCleanup, gpuContext.clear());

    ////

    gpuPropertiesCleanup.cancel();
    gpuContextCleanup.cancel();
}

//================================================================
//
// GpuAllocatorJoinContextThunk
//
// GpuMemoryAllocator + GpuContext ==> AllocatorInterface
//
//================================================================

template <typename AddrU>
class GpuAllocatorJoinContextThunk : public AllocatorInterface<AddrU>
{

public:

    void alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdParsNull)
    {
        base->alloc(context, size, alignment, owner, result, stdPassNullThru);
    }

    void setup(GpuMemoryAllocator<AddrU>& base, const GpuContext& context)
    {
        this->base = &base;
        this->context = context;
    }

private:

    GpuMemoryAllocator<AddrU>* base;
    GpuContext context;

};

//================================================================
//
// GpuShellImpl::serialize
//
//================================================================

void GpuShellImpl::serialize(const CfgSerializeKit& kit, bool hotkeys)
{
    {
        CFG_NAMESPACE("GPU Shell");

        gpuEnqueueModeCycle.serialize(kit, STR("GPU Skip Mode"), hotkeys ? STR("Ctrl+Alt+Shift+.") : STR(""));
        gpuEnqueueModeVar = (gpuEnqueueModeVar + gpuEnqueueModeCycle) % 3;

        gpuCoverageModeVar.serialize(kit, STR("GPU Coverage Mode"), hotkeys ? STR("Ctrl+.") : STR(""));
        coverageQueueCapacity.serialize(kit, STR("Coverage Queue Capacity"));
    }
}

//================================================================
//
// GpuShellImpl::execCyclicShell
//
//================================================================

void GpuShellImpl::execCyclicShell(GpuShellTarget& app, stdPars(ExecCyclicToolkit))
{
    GpuInitKit initKit = kit.gpuInitApi.getKit();

    //----------------------------------------------------------------
    //
    // Set the current thread GPU context.
    //
    //----------------------------------------------------------------

    GpuThreadContextSave contextSave;

    kit.gpuInitApi.threadContextSet(kit.gpuCurrentContext, contextSave, stdPass);
    REMEMBER_CLEANUP_ERROR_BLOCK(kit.gpuInitApi.threadContextRestore(contextSave, stdPassNc));

    //----------------------------------------------------------------
    //
    // Allocators.
    //
    //----------------------------------------------------------------

    GpuAllocatorJoinContextThunk<CpuAddrU> cpuFlatAllocator;
    cpuFlatAllocator.setup(initKit.gpuMemoryAllocation.cpuAllocator(), kit.gpuCurrentContext);

    GpuAllocatorJoinContextThunk<GpuAddrU> gpuFlatAllocator;
    gpuFlatAllocator.setup(initKit.gpuMemoryAllocation.gpuAllocator(), kit.gpuCurrentContext);

    //----------------------------------------------------------------
    //
    // Benchmark mode!
    //
    //----------------------------------------------------------------

    kit.gpuExecApi.setEnqueueMode(GpuEnqueueMode(gpuEnqueueModeVar()));
    kit.gpuExecApi.setCoverageMode(GpuCoverageMode(gpuCoverageModeVar()));

    ////

    if (gpuEnqueueModeVar() != GpuEnqueueNormal)
        printMsg(kit.localLog, gpuEnqueueModeVar() == GpuEnqueueEmptyKernel ? STR("GPU Empty Kernel Mode") : STR("GPU Call Skipping"), msgWarn);

    ////

#if HEXLIB_PLATFORM == 1

    if_not (gpuCoverageModeVar() == GpuCoverageActive)
        kit.gpuInitApi.coverageDeinit(kit.gpuCurrentStream);
    else
    {
        printMsg(kit.localLog, STR("GPU Coverage Mode"), msgWarn);
        kit.gpuInitApi.coverageInit(kit.gpuCurrentStream, coverageQueueCapacity, stdPass);
    }

#endif

    ////

    GpuExecKit execKit = kit.gpuExecApi.getKit();

    //
    //
    //

    GpuShellKit execAppKit = kitCombine
    (
        kit,
        initKit,
        execKit,
        GpuSystemAllocatorsKit{cpuFlatAllocator, gpuFlatAllocator, kit.gpuInitApi.getKit().gpuTextureAlloc}
    );

    //----------------------------------------------------------------
    //
    // Core
    //
    //----------------------------------------------------------------

    app(stdPassKit(execAppKit));

    //----------------------------------------------------------------
    //
    // Checks
    //
    //----------------------------------------------------------------

    if (kit.gpuInitApi.textureAllocCount)
        printMsg(kit.localLog, STR("Texture Reallocation: %0 times"), kit.gpuInitApi.textureAllocCount, msgWarn);

    ////

#if HEXLIB_PLATFORM == 1

    if (kit.gpuInitApi.coverageGetSyncFlag(kit.gpuCurrentStream))
    {
        printMsg(kit.localLog, STR("GPU Coverage: Sync Happened!"), kit.gpuInitApi.textureAllocCount, msgWarn);
        kit.gpuInitApi.coverageClearSyncFlag(kit.gpuCurrentStream);
    }

#endif
}

//----------------------------------------------------------------

}
