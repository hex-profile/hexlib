#include "gpuShell.h"

#include "storage/rememberCleanup.h"
#include "userOutput/printMsg.h"
#include "userOutput/paramMsg.h"

namespace gpuShell {

//================================================================
//
// GpuContextHelper::createContext
//
//================================================================

bool GpuContextHelper::createContext(GpuProperties& gpuProperties, GpuContextOwner& gpuContext, stdPars(InitKit))
{
    stdBegin;

    int32 gpuDeviceCount = 0;
    require(kit.gpuInitialization.getDeviceCount(gpuDeviceCount, stdPass));

    REQUIRE_EX(gpuDeviceCount >= 1, printMsg(kit.msgLog, STR("No GPU devices found"), msgErr));

    //
    // Use selected GPU device
    //

    gpuDeviceIndex = clampRange(gpuDeviceIndex(), 0, gpuDeviceCount-1);

    require(kit.gpuInitialization.getProperties(gpuDeviceIndex, gpuProperties, stdPass));
    REMEMBER_CLEANUP1_EX(gpuPropertiesCleanup, gpuProperties.clear(), GpuProperties&, gpuProperties);

    //
    // Context
    //

    void* baseContext = 0;
    require(kit.gpuContextCreation.createContext(gpuDeviceIndex, gpuContext, baseContext, stdPass));
    REMEMBER_CLEANUP1_EX(gpuContextCleanup, gpuContext.clear(), GpuContextOwner&, gpuContext);

    ////

    require(kit.gpuContextCreation.setThreadContext(gpuContext, stdPass));

    ////

    gpuPropertiesCleanup.cancel();
    gpuContextCleanup.cancel();

    stdEnd;
}

//================================================================
//
// GpuAllocatorJoinContextThunk
//
// GpuMemoryAllocator + GpuContext ==> FlatMemoryAllocator
//
//================================================================

template <typename AddrU>
class GpuAllocatorJoinContextThunk : public FlatMemoryAllocator<AddrU>
{

public:

    bool alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdNullPars)
        {return base->alloc(context, size, alignment, owner, result, stdNullPassThru);}

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

void GpuShellImpl::serialize(const ModuleSerializeKit& kit)
{
    {
        CFG_NAMESPACE_MODULE("GPU Shell");

        gpuEnqueueModeCycle.serialize(kit, STR("GPU Skip Mode"), STR("Alt+."));
        gpuEnqueueModeVar = (gpuEnqueueModeVar + gpuEnqueueModeCycle) % 3;

        gpuCoverageModeVar.serialize(kit, STR("GPU Coverage Mode"), STR("Ctrl+."));
        coverageQueueCapacity.serialize(kit, STR("Coverage Queue Capacity"));
    }
}

//================================================================
//
// GpuShellImpl::execCyclicShell
//
//================================================================

bool GpuShellImpl::execCyclicShell(GpuShellTarget& app, stdPars(ExecCyclicToolkit))
{
    stdBegin;

    ////

    GpuInitKit initKit = kit.gpuInitApi.getKit();

    ////

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
        require(kit.gpuInitApi.coverageInit(kit.gpuCurrentStream, coverageQueueCapacity, stdPass));
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
        GpuSystemAllocatorsKit(cpuFlatAllocator, gpuFlatAllocator, kit.gpuInitApi.getKit().gpuTextureAlloc)
    );

    //----------------------------------------------------------------
    //
    // Core
    //
    //----------------------------------------------------------------

    require(app.exec(stdPassKit(execAppKit)));

    //----------------------------------------------------------------
    //
    // Checks
    //
    //----------------------------------------------------------------

    if (kit.gpuInitApi.textureAllocCount)
        printMsg(kit.localLog, STR("Texture Reallocation: %0 times"), kit.gpuInitApi.textureAllocCount, msgWarn);

    ////

#if HEXLIB_PLATFORM != 0

    if (kit.gpuInitApi.coverageGetSyncFlag(kit.gpuCurrentStream))
    {
        printMsg(kit.localLog, STR("GPU Coverage: Sync Happened!"), kit.gpuInitApi.textureAllocCount, msgWarn);
        kit.gpuInitApi.coverageClearSyncFlag(kit.gpuCurrentStream);
    }

#endif

    ////

    stdEnd;
}

//----------------------------------------------------------------

}
