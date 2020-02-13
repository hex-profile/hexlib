#include "minimalShell.h"

#include "cfgTools/boolSwitch.h"
#include "errorLog/debugBreak.h"
#include "gpuLayer/gpuCallsProhibition.h"
#include "gpuShell/gpuShell.h"
#include "imageConsole/gpuImageConsole.h"
#include "kits/setBusyStatus.h"
#include "storage/classThunks.h"
#include "storage/disposableObject.h"
#include "storage/rememberCleanup.h"
#include "userOutput/paramMsg.h"
#include "userOutput/printMsg.h"
#include "memController/memoryUsageReport.h"

namespace minimalShell {

using namespace gpuShell;

//================================================================
//
// ProcessEnrichedKit
// ProcessGpuKit
//
//================================================================

using ProcessEnrichedKit = KitCombine<ProcessKit, ProfilerKit, UserPointKit, SetBusyStatusKit, DisplayParamsKit, AlternativeVersionKit, VerbosityKit, GpuImageConsoleKit>;
using ProcessGpuKit = KitCombine<ProcessEnrichedKit, GpuShellKit>;

//================================================================
//
// EngineModuleReallocThunk
//
//================================================================

class EngineModuleReallocThunk : public MemControllerReallocTarget
{

    using BaseKit = ProcessGpuKit;

public:

    bool reallocValid() const
        {return engineModule.reallocValid();}

public:

    stdbool realloc(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseKit);
        BaseKit joinKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

        return engineModule.realloc(stdPassThruKit(kitCombine(kit, joinKit)));
    }

public:

    inline EngineModuleReallocThunk(EngineModule& engineModule, const BaseKit& baseKit)
        : engineModule(engineModule), baseKit(baseKit) {}

private:

    EngineModule& engineModule;
    BaseKit const baseKit;

};

//================================================================
//
// EngineModuleProcessThunk
//
//================================================================

class EngineModuleProcessThunk : public MemControllerProcessTarget
{

public:

    using BaseKit = KitCombine<ProcessGpuKit, PipeControlKit>;

public:

    stdbool process(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseKit);
        BaseKit joinKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

        return engineModule.process(stdPassThruKit(kitCombine(kit, joinKit)));
    }

public:

    inline EngineModuleProcessThunk(EngineModule& engineModule, const BaseKit& baseKit)
        : engineModule(engineModule), baseKit(baseKit) {}

private:

    EngineModule& engineModule;
    BaseKit const baseKit;

};

//================================================================
//
// processWithGpu
//
//================================================================

stdbool processWithGpu(EngineModule& engineModule, MemController& engineMemory, stdPars(ProcessGpuKit))
{

    //----------------------------------------------------------------
    //
    // Engine module state memory
    //
    //----------------------------------------------------------------

    MemoryUsage engineStateUsage;
    ReallocActivity engineStateActivity;

    {
        EngineModuleReallocThunk engineModuleThunk(engineModule, kit);
        require(engineMemory.handleStateRealloc(engineModuleThunk, kit, engineStateUsage, engineStateActivity, stdPass));
    }

    REQUIRE(engineStateActivity.fastAllocCount <= 1);
    REQUIRE(engineStateActivity.sysAllocCount <= 1);

    //----------------------------------------------------------------
    //
    // Count engine module temp memory
    //
    //----------------------------------------------------------------

    MemoryUsage engineTempUsage;

    {
        //
        // Pipe control on memory counting stage: advance 1 frame
        //

        PipeControl pipeControl(0, false);
        auto kitEx = kitCombine(kit, PipeControlKit(pipeControl));

        ////

        EngineModuleProcessThunk engineModuleThunk(engineModule, kitEx);
        require(engineMemory.processCountTemp(engineModuleThunk, engineTempUsage, stdPassKit(kitEx)));
    }

    //----------------------------------------------------------------
    //
    // Reallocate temp memory pools (if necessary).
    //
    //----------------------------------------------------------------

    ReallocActivity engineTempActivity;
    require(engineMemory.handleTempRealloc(engineTempUsage, kit, engineTempActivity, stdPass));

    REQUIRE(engineTempActivity.fastAllocCount <= 1);
    REQUIRE(engineTempActivity.sysAllocCount <= 1);
    REQUIRE(engineStateActivity.fastAllocCount <= 1);
    REQUIRE(engineStateActivity.sysAllocCount <= 1);

    if (uncommonActivity(engineStateActivity, engineTempActivity))
        memoryUsageReport(STR("Engine"), engineStateUsage, engineTempUsage, engineStateActivity, engineTempActivity, stdPass);

    //----------------------------------------------------------------
    //
    // Execute process with memory distribution.
    //
    //----------------------------------------------------------------

    {
        //
        // Pipe control on execution stage: advance 0 frames (rollback 1 frame)
        //

        PipeControl pipeControl(1, false);
        auto kitEx = kitCombine(kit, PipeControlKit(pipeControl));

        ////

        EngineModuleProcessThunk engineModuleThunk(engineModule, kitEx);
        MemoryUsage actualEngineTempUsage;

        require(engineMemory.processAllocTemp(engineModuleThunk, kitEx, actualEngineTempUsage, stdPass));

        CHECK(actualEngineTempUsage == engineTempUsage);
    }

    ////

    returnTrue;
}

//================================================================
//
// GpuShellExecAppImpl
//
//================================================================

template <typename BaseKit>
class GpuShellExecAppImpl : public GpuShellTarget
{

public:

    stdbool exec(stdPars(GpuShellKit))
        {return processWithGpu(engineModule, engineMemory, stdPassThruKit(kitCombine(baseKit, kit)));}

    inline GpuShellExecAppImpl(EngineModule& engineModule, MemController& engineMemory, const BaseKit& baseKit)
        : engineModule(engineModule), engineMemory(engineMemory), baseKit(baseKit) {}

private:

    EngineModule& engineModule; 
    MemController& engineMemory;
    BaseKit const baseKit;

};

//================================================================
//
// MinimalShellImpl
//
//================================================================

class MinimalShellImpl : public CfgSerialization
{

public:

    void serialize(const CfgSerializeKit& kit);

public:

    stdbool init(stdPars(InitKit));

public:

    stdbool process(EngineModule& engineModule, MemController& engineMemory, stdPars(ProcessKit));

private:

    bool initialized = false;

    //
    // GPU layer
    //

    GpuContextHelper gpuContextHelper;
    GpuProperties gpuProperties;
    GpuContextOwner gpuContext;
    GpuStreamOwner gpuStream;

    GpuShellImpl gpuShell;

};

//================================================================
//
// MinimalShellImpl::serialize
//
//================================================================

void MinimalShellImpl::serialize(const CfgSerializeKit& kit)
{
    {
        CFG_NAMESPACE("Minimal Shell");
        gpuShell.serialize(kit);
        gpuContextHelper.serialize(kit);
    }
}

//================================================================
//
// MinimalShellImpl::init
//
//================================================================

stdbool MinimalShellImpl::init(stdPars(InitKit))
{
    initialized = false;

    //
    // GPU init
    //

    GpuInitApiImpl gpuInitApi(kit);
    require(gpuInitApi.initialize(stdPass));
    GpuInitKit gpuInitKit = gpuInitApi.getKit();

    ////

    require(gpuContextHelper.createContext(gpuProperties, gpuContext, stdPassKit(kitCombine(kit, gpuInitKit))));
    REMEMBER_CLEANUP1_EX(gpuContextCleanup, gpuContext.clear(), GpuContextOwner&, gpuContext);

    ////

    void* baseStream = 0;
    require(gpuInitKit.gpuStreamCreation.createStream(gpuContext, true, gpuStream, baseStream, stdPass));

    //
    // Record success
    //

    gpuContextCleanup.cancel();
    initialized = true;

    returnTrue;
}

//================================================================
//
// MinimalShellImpl::process
//
//================================================================

stdbool MinimalShellImpl::process(EngineModule& engineModule, MemController& engineMemory, stdPars(ProcessKit))
{
    REQUIRE_EX(initialized, printMsg(kit.localLog, STR("Initialization failed, processing is disabled"), msgWarn));

    //----------------------------------------------------------------
    //
    // Proceed
    //
    //----------------------------------------------------------------

    ProfilerKit profilerKit(nullptr);

    ////

    UserPoint userPoint{false, point(0), false, false};

    ////

    AlternativeVersionKit alternativeVersionKit(false);

    ////

    DisplayedRangeIndex viewIndex(0);
    DisplayedRangeIndex rangeInex(0);
    DisplayedCircularIndex circularIndex(0);
    DisplayedRangeIndex stageIndex(0);
    DisplayedCircularIndex channelIndex(0);

    DisplayParams displayParams
    {
        true,
        1.f,
        point(0),
        false,
        viewIndex,
        rangeInex,
        rangeInex,
        circularIndex,
        stageIndex,
        channelIndex
    };

    ////

    GpuInitApiImpl gpuInitApi(kit);
    GpuExecApiImpl gpuExecApi(kitCombine(kit, profilerKit));

    ////

    SetBusyStatusNull setBusyStatus;

    ////

    GpuImageConsoleNull gpuImageConsole;
    GpuImageConsoleKit gpuImageConsoleKit{gpuImageConsole};

    ////

    auto kitEx = kitCombine
    (
        kit,
        profilerKit,
        UserPointKit{userPoint},
        SetBusyStatusKit{setBusyStatus},
        DisplayParamsKit{displayParams},
        alternativeVersionKit,
        VerbosityKit{Verbosity::On},
        gpuImageConsoleKit,
        GpuApiImplKit{gpuInitApi, gpuExecApi},
        GpuPropertiesKit{gpuProperties},
        GpuCurrentContextKit{gpuContext},
        GpuCurrentStreamKit{gpuStream}
    );

    ////

    GpuShellExecAppImpl<ProcessEnrichedKit> gpuShellExecApp(engineModule, engineMemory, kitEx);
    require(gpuShell.execCyclicShell(gpuShellExecApp, stdPassKit(kitEx)));

    ////

    returnTrue;
}

//================================================================
//
// Thunks
//
//================================================================

CLASSTHUNK_CONSTRUCT_DESTRUCT(MinimalShell)
CLASSTHUNK_VOID1(MinimalShell, serialize, const CfgSerializeKit&)
CLASSTHUNK_BOOL_STD0(MinimalShell, init, InitKit)
CLASSTHUNK_BOOL_STD2(MinimalShell, process, EngineModule&, MemController&, ProcessKit)

//----------------------------------------------------------------

}
