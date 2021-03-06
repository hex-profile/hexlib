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
#include "configFile/cfgSimpleString.h"
#include "userOutput/printMsgEx.h"
#include "gpuImageConsoleImpl/gpuImageConsoleImpl.h"
#include "gpuBaseConsoleByCpu/gpuBaseConsoleByCpu.h"
#include "profilerShell/profilerShell.h"

namespace minimalShell {

using namespace gpuShell;

//================================================================
//
// MinimalShellImpl
//
//================================================================

class MinimalShellImpl : public MinimalShell, public Settings
{

public:

    virtual Settings& settings() {return *this;}

public:

    virtual void setImageSavingActive(bool active)
        {bmpConsole->setActive(active);}

    virtual void setImageSavingDir(const CharType* dir)
        {bmpConsole->setDir(dir);}

    virtual void setImageSavingLockstepCounter(uint32 counter)
        {bmpConsole->setLockstepCounter(counter);}

    virtual const CharType* getImageSavingDir() const
        {return bmpConsole->getOutputDir();}

public:

    void serialize(const CfgSerializeKit& kit);

public:

    bool isInitialized() const {return initialized;}

    stdbool init(stdPars(InitKit));

public:

    stdbool processEntry(stdPars(ProcessEntryKit));

public:

    bool profilingActive() const 
    {
        return profilerShell.profilingActive();
    }

    stdbool profilingReport(stdPars(ReportKit))
    {
        REQUIRE(initialized);
        return profilerShell.makeReport(gpuProperties.totalThroughput, stdPass);
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

private:

    using ProcessWithProfilerKit = KitCombine<ProcessEntryKit, ProfilerKit>;

    stdbool processWithProfiler(stdPars(ProcessWithProfilerKit));

private:

    using ProcessWithGpuKit = KitCombine<ProcessWithProfilerKit, GpuShellKit>;

    stdbool processWithGpu(stdPars(ProcessWithGpuKit));

private:

    using ProcessWithAllocatorsKit = KitCombine<ProcessWithGpuKit, memController::FastAllocToolkit, PipeControlKit>;

    stdbool processWithAllocators(stdPars(ProcessWithAllocatorsKit));

private:

    bool initialized = false;

    //----------------------------------------------------------------
    //
    // Profiler.
    //
    //----------------------------------------------------------------

    ProfilerShell profilerShell;

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
    // Debugging support.
    //
    //----------------------------------------------------------------

    BoolSwitch<false> displayMemoryUsage;
    BoolSwitch<false> debugBreakOnErrors;

    DisplayParamsImpl displayParams;

    UniquePtr<BaseConsoleBmp> bmpConsole = BaseConsoleBmp::create();

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

        profilerShell.serialize(kit);

        gpuShell.serialize(kit);
        gpuContextHelper.serialize(kit);

        displayMemoryUsage.serialize(kit, STR("Display Memory Usage"));
        debugBreakOnErrors.serialize(kit, STR("Debug Break On Errors"));

        {
            CFG_NAMESPACE("Display Params");

            bool unused = false;
            displayParams.serialize(kit, unused);
        }

        {
            CFG_NAMESPACE("Saving BMP Files");
            bmpConsole->serialize(kit);
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
    // Profiler
    //

    require(profilerShell.init(stdPass));
    REMEMBER_CLEANUP_EX(profilerCleanup, profilerShell.deinit());

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
    profilerCleanup.cancel();
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

    auto originalKit = kit;
    auto kit = kitReplace(originalKit, kitCombine(msgLogKit, errorLogKit, errorLogExKit));

    //----------------------------------------------------------------
    //
    // ProfilerTargetThunk
    //
    //----------------------------------------------------------------

    class ProfilerTargetThunk : public ProfilerTarget
    {

    public:

        stdbool process(stdPars(ProfilerKit))
            {return base.processWithProfiler(stdPassThruKit(kitCombine(kit, baseKit)));}

        inline ProfilerTargetThunk(MinimalShellImpl& base, const ProcessEntryKit& baseKit)
            : base(base), baseKit(baseKit) {}

    private:

        MinimalShellImpl& base;
        ProcessEntryKit baseKit;

    };

    //----------------------------------------------------------------
    //
    // Profiler shell.
    //
    //----------------------------------------------------------------

    ProfilerTargetThunk profilerTarget(*this, kit);
    require(profilerShell.process(profilerTarget, gpuProperties.totalThroughput, stdPass));

    ////

    stdScopedEnd;
}

//================================================================
//
// MinimalShellImpl::processWithProfiler
//
//================================================================

stdbool MinimalShellImpl::processWithProfiler(stdPars(ProcessWithProfilerKit))
{

    //----------------------------------------------------------------
    //
    // GPU shell.
    //
    //----------------------------------------------------------------

    auto baseKit = kit;

    ////

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

    returnTrue;
}

//================================================================
//
// MinimalShellImpl::processWithGpu
//
//================================================================

stdbool MinimalShellImpl::processWithGpu(stdPars(ProcessWithGpuKit))
{

    kit.sysAllocHappened = false;

    //----------------------------------------------------------------
    //
    // Set the current thread GPU context.
    //
    //----------------------------------------------------------------

    require(kit.gpuContextCreation.setThreadContext(kit.gpuCurrentContext, stdPass));

    //----------------------------------------------------------------
    //
    // Give GPU control to the profiler.
    //
    //----------------------------------------------------------------

    ProfilerDeviceKit profilerDeviceKit = kit;
    profilerShell.setDeviceControl(&profilerDeviceKit);
    REMEMBER_CLEANUP(profilerShell.setDeviceControl(nullptr));

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

    ////

    if (engineStateActivity.sysAllocCount)
        kit.sysAllocHappened = true;

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

    ////

    if (engineTempActivity.sysAllocCount)
        kit.sysAllocHappened = true;

    ////

    if (displayMemoryUsage || uncommonActivity(engineStateActivity, engineTempActivity))
        memoryUsageReport(STR("Engine"), engineStateUsage, engineTempUsage, engineStateActivity, engineTempActivity, stdPass);

    //----------------------------------------------------------------
    //
    // Pure counting mode.
    //
    //----------------------------------------------------------------

    if_not (kit.runExecutionPhase)
        returnTrue;

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
    stdScopedBegin;

    //----------------------------------------------------------------
    //
    // Disabled kits.
    //
    //----------------------------------------------------------------

    SetBusyStatusNull setBusyStatus;

    auto oldKit = kit;

    auto kit = kitCombine
    (
        oldKit,
        SetBusyStatusKit{setBusyStatus},
        VerbosityKit{Verbosity::On}
    );

    //----------------------------------------------------------------
    //
    // CPU base console.
    //
    //----------------------------------------------------------------

    BaseImageConsole* baseConsole = kit.baseImageConsole;

    BaseImageConsoleNull baseConsoleNull;

    if_not (baseConsole)
        baseConsole = &baseConsoleNull;

    ////

    BaseVideoOverlay* baseOverlay = kit.baseVideoOverlay;

    BaseVideoOverlayNull baseOverlayNull;

    if_not (baseOverlay)
        baseOverlay = &baseOverlayNull;

    ////

    BaseConsoleBmpThunk bmpThunk(*bmpConsole, *baseConsole, *baseOverlay, kit);

    if (bmpConsole->active())
    {
        printMsgL(kit, STR("Image Saving: Files are saved to %0"), bmpConsole->getOutputDir());
        baseConsole = &bmpThunk; 
        baseOverlay = &bmpThunk;
    }

    //----------------------------------------------------------------
    //
    // GPU image console.
    //
    //----------------------------------------------------------------

    using namespace gpuImageConsoleImpl;

    ////

    GpuBaseConsoleProhibitThunk gpuBaseConsoleDisabled(kit);

    GpuBaseConsoleByCpuThunk gpuBaseConsoleEnabled(*baseConsole, *baseOverlay, kit);

    GpuBaseConsole* gpuBaseConsole = &gpuBaseConsoleDisabled;

    if (kit.verbosity >= Verbosity::On)
        gpuBaseConsole = &gpuBaseConsoleEnabled;

    ////

    GpuImageConsoleThunk gpuImageConsole(*gpuBaseConsole, displayParams.displayMode(), displayParams.vectorMode(), kit);

    GpuImageConsoleKit gpuImageConsoleKit{gpuImageConsole};

    //----------------------------------------------------------------
    //
    // Display params and kit.
    //
    //----------------------------------------------------------------

    DisplayParamsThunk displayParamsThunk{point(1), displayParams}; // Screen size not supported well.

    ////

    auto kitEx = kitCombine
    (
        kit,
        gpuImageConsoleKit,
        displayParamsThunk.getKit()
    );

    ////

    require(kit.engineModule.process(stdPassKit(kitEx)));

    ////

    stdScopedEnd;
}

//----------------------------------------------------------------

}
