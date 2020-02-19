#include "minimalShell.h"

#include "cfgTools/boolSwitch.h"
#include "errorLog/debugBreak.h"
#include "formattedOutput/errorBreakThunks.h"
#include "gpuLayer/gpuCallsProhibition.h"
#include "gpuShell/gpuShell.h"
#include "imageConsole/gpuImageConsole.h"
#include "kits/setBusyStatus.h"
#include "memController/memoryUsageReport.h"
#include "storage/classThunks.h"
#include "storage/disposableObject.h"
#include "storage/rememberCleanup.h"
#include "userOutput/paramMsg.h"
#include "userOutput/printMsg.h"
#include "displayParamsImpl/displayParamsImpl.h"
#include "baseConsoleBmp/baseConsoleBmp.h"

namespace minimalShell {

using namespace gpuShell;

//================================================================
//
// MinimalShellImpl
//
//================================================================

class MinimalShellImpl : public MinimalShell
{

public:

    void serialize(const CfgSerializeKit& kit);

public:

    stdbool init(stdPars(InitKit));

public:

    using ProcessEnrichedKit = KitCombine<ProcessEntryKit, ProfilerKit>;

    stdbool processEntry(stdPars(ProcessEntryKit));

private:

    using ProcessWithGpuKit = KitCombine<ProcessEnrichedKit, GpuShellKit>;

    stdbool processWithGpu(stdPars(ProcessWithGpuKit));

private:

    using ProcessWithAllocatorsKit = KitCombine<ProcessWithGpuKit, memController::FastAllocToolkit, PipeControlKit>;

    stdbool processWithAllocators(stdPars(ProcessWithAllocatorsKit));

private:

    bool initialized = false;

    //----------------------------------------------------------------
    //
    // GPU layer
    //
    //----------------------------------------------------------------

    GpuContextHelper gpuContextHelper;
    GpuProperties gpuProperties;
    GpuContextOwner gpuContext;
    GpuStreamOwner gpuStream;

    GpuShellImpl gpuShell;

    //----------------------------------------------------------------
    //
    // Options.
    //
    //----------------------------------------------------------------

    BoolSwitch<false> displayMemoryUsage;
    BoolSwitch<false> debugBreakOnErrors;

    DisplayParamsImpl displayParams;

};

//----------------------------------------------------------------

UniquePtr<MinimalShell> MinimalShell::create()
{
    return makeUnique<MinimalShellImpl>();
}

//================================================================
//
// MinimalShellImpl::serialize
//
//================================================================

void MinimalShellImpl::serialize(const CfgSerializeKit& kit)
{
    {
        CFG_NAMESPACE("~Shell");

        gpuShell.serialize(kit);
        gpuContextHelper.serialize(kit);

        displayMemoryUsage.serialize(kit, STR("Display Memory Usage"));
        debugBreakOnErrors.serialize(kit, STR("Debug Break On Errors"));

        {
            CFG_NAMESPACE("Display Params");

            bool unused = false;
            displayParams.serialize(kit, unused);
        }
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
// MinimalShellImpl::processEntry
//
//================================================================

stdbool MinimalShellImpl::processEntry(stdPars(ProcessEntryKit))
{
    stdScopedBegin;

    REQUIRE_EX(initialized, printMsg(kit.localLog, STR("Initialization failed, processing is disabled"), msgWarn));

    //----------------------------------------------------------------
    //
    // Intercept error output and add debug break functionality.
    //
    //----------------------------------------------------------------

    MsgLogBreakShell msgLog(kit.msgLog, debugBreakOnErrors);
    MsgLogKit msgLogKit(msgLog);

    ErrorLogBreakShell errorLog(kit.errorLog, debugBreakOnErrors);
    ErrorLogKit errorLogKit(errorLog);

    ErrorLogExBreakShell errorLogEx(kit.errorLogEx, debugBreakOnErrors);
    ErrorLogExKit errorLogExKit(errorLogEx);

    auto errorBreakKit = kitReplace(kit, kitCombine(msgLogKit, errorLogKit, errorLogExKit));

    //----------------------------------------------------------------
    //
    // Profiler.
    //
    //----------------------------------------------------------------

    ProfilerKit profilerKit(nullptr);
    auto baseKit = kitCombine(errorBreakKit, profilerKit);

    //----------------------------------------------------------------
    //
    // Proceed
    //
    //----------------------------------------------------------------

    auto gpuShellExec = [this, &baseKit] (stdPars(GpuShellKit))
    {
        return processWithGpu(stdPassThruKit(kitCombine(baseKit, kit)));
    };

    ////

    GpuInitApiImpl gpuInitApi(baseKit);
    GpuExecApiImpl gpuExecApi(baseKit);

    auto kitEx = kitCombine
    (
        baseKit,
        GpuApiImplKit{gpuInitApi, gpuExecApi},
        GpuPropertiesKit{gpuProperties},
        GpuCurrentContextKit{gpuContext},
        GpuCurrentStreamKit{gpuStream}
    );

    require(gpuShell.execCyclicShellLambda(gpuShellExec, stdPassKit(kitEx)));

    ////

    stdScopedEnd;
}

//================================================================
//
// MinimalShellImpl::processWithGpu
//
//================================================================

stdbool MinimalShellImpl::processWithGpu(stdPars(ProcessWithGpuKit))
{

    //----------------------------------------------------------------
    //
    // ReallocThunk
    //
    //----------------------------------------------------------------

    class ReallocThunk : public MemControllerReallocTarget
    {
        using BaseKit = ProcessWithGpuKit;

        bool reallocValid() const
        {
            return engineModule.reallocValid();
        }

        stdbool realloc(stdPars(memController::FastAllocToolkit))
        {
            GpuProhibitedExecApiThunk prohibitedApi(baseKit);
            BaseKit joinKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

            return engineModule.realloc(stdPassThruKit(kitCombine(kit, joinKit)));
        }

        CLASS_CONTEXT(ReallocThunk, ((EngineModule&, engineModule)) ((BaseKit, baseKit)));
    };

    //----------------------------------------------------------------
    //
    // Engine module state memory
    //
    //----------------------------------------------------------------

    MemoryUsage engineStateUsage;
    ReallocActivity engineStateActivity;

    ////

    {
        ReallocThunk reallocThunk(kit.engineModule, kit);
        require(kit.engineMemory.handleStateRealloc(reallocThunk, kit, engineStateUsage, engineStateActivity, stdPass));
    }

    ////

    REQUIRE(engineStateActivity.fastAllocCount <= 1);
    REQUIRE(engineStateActivity.sysAllocCount <= 1);

    //----------------------------------------------------------------
    //
    // ProcessThunk
    //
    //----------------------------------------------------------------

    class ProcessThunk : public MemControllerProcessTarget
    {
        using BaseKit = KitCombine<ProcessWithGpuKit, PipeControlKit>;

        stdbool process(stdPars(memController::FastAllocToolkit))
        {
            GpuProhibitedExecApiThunk prohibitedApi(baseKit);
            BaseKit joinKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

            return shell.processWithAllocators(stdPassThruKit(kitCombine(kit, joinKit)));
        }

        CLASS_CONTEXT(ProcessThunk, ((MinimalShellImpl&, shell)) ((BaseKit, baseKit)));
    };

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

        ProcessThunk processThunk{*this, kitEx};
        require(kit.engineMemory.processCountTemp(processThunk, engineTempUsage, stdPassKit(kitEx)));
    }

    //----------------------------------------------------------------
    //
    // Reallocate temp memory pools (if necessary).
    //
    //----------------------------------------------------------------

    ReallocActivity engineTempActivity;
    require(kit.engineMemory.handleTempRealloc(engineTempUsage, kit, engineTempActivity, stdPass));

    REQUIRE(engineTempActivity.fastAllocCount <= 1);
    REQUIRE(engineTempActivity.sysAllocCount <= 1);
    REQUIRE(engineStateActivity.fastAllocCount <= 1);
    REQUIRE(engineStateActivity.sysAllocCount <= 1);

    if (displayMemoryUsage || uncommonActivity(engineStateActivity, engineTempActivity))
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

        ProcessThunk processThunk{*this, kitEx};
        MemoryUsage actualEngineTempUsage;

        require(kit.engineMemory.processAllocTemp(processThunk, kitEx, actualEngineTempUsage, stdPass));

        CHECK(actualEngineTempUsage == engineTempUsage);
    }

    ////

    returnTrue;
}

//================================================================
//
// MinimalShellImpl::processWithAllocators
//
//================================================================

stdbool MinimalShellImpl::processWithAllocators(stdPars(ProcessWithAllocatorsKit))
{

    SetBusyStatusNull setBusyStatus;
    UserPoint userPoint{false, point(0), false, false};

    ////

    GpuImageConsoleNull gpuImageConsole;
    GpuImageConsoleKit gpuImageConsoleKit{gpuImageConsole};

    ////

    DisplayParamsThunk displayParamsThunk{point(1), displayParams}; // Screen size not supported well.

    ////

    auto kitEx = kitCombine
    (
        kit,
        SetBusyStatusKit{setBusyStatus},
        UserPointKit{userPoint},
        VerbosityKit{Verbosity::On},
        gpuImageConsoleKit,
        displayParamsThunk.getKit()
    );

    ////

    require(kit.engineModule.process(stdPassKit(kitEx)));

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
