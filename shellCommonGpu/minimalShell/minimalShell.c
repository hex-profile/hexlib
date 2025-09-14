#include "minimalShell.h"

#include "baseConsoleBmp/baseConsoleBmp.h"
#include "cfgTools/boolSwitch.h"
#include "cfgTools/cfgSimpleString.h"
#include "displayParamsImpl/displayParamsImpl.h"
#include "errorLog/debugBreak.h"
#include "formattedOutput/errorBreakThunks.h"
#include "gpuBaseConsoleByCpu/gpuBaseConsoleByCpu.h"
#include "gpuImageConsoleImpl/gpuImageConsoleImpl.h"
#include "gpuLayer/gpuCallsProhibition.h"
#include "gpuShell/gpuShell.h"
#include "imageConsole/gpuImageConsole.h"
#include "kits/setBusyStatus.h"
#include "memController/memoryUsageReport.h"
#include "profilerShell/profilerShell.h"
#include "storage/classThunks.h"
#include "storage/optionalObject.h"
#include "storage/rememberCleanup.h"
#include "userOutput/paramMsg.h"
#include "userOutput/printMsg.h"
#include "userOutput/printMsgEx.h"
#include "timer/changesMonitor.h"

namespace minimalShell {

using namespace gpuShell;

//================================================================
//
// MinimalShellImpl
//
//================================================================

class MinimalShellImpl : public MinimalShell, public Settings
{

    //----------------------------------------------------------------
    //
    // Settings.
    //
    //----------------------------------------------------------------

    virtual Settings& settings() {return *this;}

    ////

    virtual void setGpuContextMaintainer(bool value)
        {gpuContextMaintainer = value;}

    virtual void setGpuShellHotkeys(bool value)
        {gpuShellHotkeys = value;}

    virtual void setProfilerShellHotkeys(bool value)
        {profilerShellHotkeys = value;}

    virtual void setDisplayParamsHotkeys(bool value)
        {displayParamsHotkeys = value;}

    virtual void setBmpConsoleHotkeys(bool value)
        {bmpConsoleHotkeys = value;}

    ////

    virtual void setImageSavingActive(bool active)
        {bmpConsole->setActive(active);}

    virtual void setImageSavingDir(const CharType* dir)
        {bmpConsole->setDir(dir);}

    virtual void setImageSavingLockstepCounter(uint32 counter)
        {bmpConsole->setLockstepCounter(counter);}

    virtual const CharType* getImageSavingDir() const
        {return bmpConsole->getOutputDir();}

    ////

    void serialize(const CfgSerializeKit& kit);

    //----------------------------------------------------------------
    //
    // Init.
    //
    //----------------------------------------------------------------

    void init(stdPars(InitKit));

    //----------------------------------------------------------------
    //
    // Processing.
    //
    //----------------------------------------------------------------

    void process(const ProcessArgs& args, stdPars(ProcessKit));

    //----------------------------------------------------------------
    //
    // Profiling.
    //
    //----------------------------------------------------------------

    bool profilingActive() const
    {
        return profilerShell.profilingActive();
    }

    void profilingReport(const GpuExternalContext* externalContext, stdPars(ReportKit))
    {
        REQUIRE(level >= Level::InitExternal);

        auto& gpuProperties = (level == Level::InitInternal) ? gpuPropertiesInternal : externalContext->gpuProperties;

        profilerShell.makeReport(gpuProperties.totalThroughput, stdPass);
    }

    //----------------------------------------------------------------
    //
    // Private funcs.
    //
    //----------------------------------------------------------------

private:

    using ProcessWithProfilerKit = KitCombine<ProcessKit, ProfilerKit>;

    void processWithProfiler(const ProcessArgs& args, stdPars(ProcessWithProfilerKit));

    ////

    using ProcessWithGpuKit = KitCombine<ProcessWithProfilerKit, GpuShellKit>;

    void processWithGpu(const ProcessArgs& args, stdPars(ProcessWithGpuKit));

    ////

    using ProcessWithAllocatorsKit = KitCombine<ProcessWithGpuKit, memController::FastAllocToolkit, PipeControlKit>;

    void processWithAllocators(const ProcessArgs& args, stdPars(ProcessWithAllocatorsKit));

    //----------------------------------------------------------------
    //
    // Status.
    //
    //----------------------------------------------------------------

    enum class Level {None, InitExternal, InitInternal};

    Level level = Level::None;

    //----------------------------------------------------------------
    //
    // Profiler.
    //
    //----------------------------------------------------------------

    bool profilerShellHotkeys = true;
    ProfilerShell profilerShell;

    //----------------------------------------------------------------
    //
    // GPU context and stream.
    //
    //----------------------------------------------------------------

    bool gpuContextMaintainer = true;

    ////

    GpuContextHelper gpuContextHelper;

    GpuProperties gpuPropertiesInternal;
    GpuContextOwner gpuContextInternal;
    GpuStreamOwner gpuStreamInternal;

    //----------------------------------------------------------------
    //
    // Gpu shell.
    //
    //----------------------------------------------------------------

    bool gpuShellHotkeys = true;
    GpuShellImpl gpuShell;

    //----------------------------------------------------------------
    //
    // Debugging support.
    //
    //----------------------------------------------------------------

    BoolSwitch displayMemoryUsage{false};
    BoolSwitch debugBreakOnErrors{false};

    bool displayParamsHotkeys = true;
    DisplayParamsImpl displayParams;
    ChangesMonitor altVersionMonitor;

    bool bmpConsoleHotkeys = true;
    UniqueInstance<BaseConsoleBmp> bmpConsole;
    baseConsoleBmp::Counter bmpCounter = 0;

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
    profilerShell.serialize(kit, profilerShellHotkeys);

    ////

    if (gpuContextMaintainer)
        gpuContextHelper.serialize(kit);

    ////

    gpuShell.serialize(kit, gpuShellHotkeys);

    ////

    displayMemoryUsage.serialize(kit, STR("Display Memory Usage"), STR("Ctrl+Shift+U"));
    debugBreakOnErrors.serialize(kit, STR("Debug Break On Errors"));

    ////

    {
        CFG_NAMESPACE("Display Params");

        bool altVersionSteady = true;
        displayParams.serialize(kit, displayParamsHotkeys, altVersionSteady);
        altVersionMonitor.touch(!altVersionSteady);
    }

    ////

    {
        CFG_NAMESPACE("Saving BMP Files");
        bmpConsole->serialize(kit, bmpConsoleHotkeys);
    }
}

//================================================================
//
// MinimalShellImpl::init
//
//================================================================

void MinimalShellImpl::init(stdPars(InitKit))
{

    //----------------------------------------------------------------
    //
    // Profiler
    //
    //----------------------------------------------------------------

    profilerShell.init(stdPass);
    REMEMBER_CLEANUP_EX(profilerCleanup, profilerShell.deinit());

    //----------------------------------------------------------------
    //
    // GPU init
    //
    //----------------------------------------------------------------

    if (gpuContextMaintainer)
    {
        GpuInitApiImpl gpuInitApi(kit);
        gpuInitApi.initialize(stdPass);
        GpuInitKit gpuInitKit = gpuInitApi.getKit();

        ////

        gpuContextHelper.createContext(gpuPropertiesInternal, gpuContextInternal, stdPassKit(kitCombine(kit, gpuInitKit)));
        REMEMBER_CLEANUP_EX(gpuContextCleanup, gpuContextInternal.clear(););

        ////

        gpuInitKit.gpuStreamCreation.createStream(gpuContextInternal, true, gpuStreamInternal, stdPass);

        gpuContextCleanup.cancel();
    }

    //----------------------------------------------------------------
    //
    // Record success
    //
    //----------------------------------------------------------------

    profilerCleanup.cancel();

    level = gpuContextMaintainer ? Level::InitInternal : Level::InitExternal;
}

//================================================================
//
// MinimalShellImpl::process
//
//================================================================

void MinimalShellImpl::process(const ProcessArgs& args, stdPars(ProcessKit))
{
    stdScopedBegin;

    REQUIRE_EX(level >= Level::InitExternal, printMsg(kit.localLog, STR("Initialization failed, processing is disabled"), msgWarn));

    //----------------------------------------------------------------
    //
    // Intercept error output and add debug break functionality.
    //
    //----------------------------------------------------------------

    MsgLogBreakShell msgLog(kit.msgLog, debugBreakOnErrors);
    MsgLogKit msgLogKit(msgLog);

    ErrorLogBreakShell errorLog(kit.errorLog, debugBreakOnErrors);
    ErrorLogKit errorLogKit(errorLog);

    MsgLogExBreakShell msgLogEx(kit.msgLogEx, debugBreakOnErrors);
    MsgLogExKit msgLogExKit(msgLogEx);

    auto originalKit = kit;
    auto kit = kitReplace(originalKit, kitCombine(msgLogKit, errorLogKit, msgLogExKit));

    //----------------------------------------------------------------
    //
    // Profiler shell.
    //
    //----------------------------------------------------------------

    auto& oldKit = kit;

    auto profilerTarget = ProfilerTarget::O | [&] (stdPars(ProfilerKit))
    {
        processWithProfiler(args, stdPassThruKit(kitCombine(oldKit, kit)));
    };

    ////

    auto& gpuProperties = (level == Level::InitInternal) ?
        gpuPropertiesInternal : args.externalContext->gpuProperties;

    profilerShell.process(profilerTarget, gpuProperties.totalThroughput, stdPass);

    ////

    stdScopedEnd;
}

//================================================================
//
// MinimalShellImpl::processWithProfiler
//
//================================================================

void MinimalShellImpl::processWithProfiler(const ProcessArgs& args, stdPars(ProcessWithProfilerKit))
{

    //----------------------------------------------------------------
    //
    // GPU shell.
    //
    //----------------------------------------------------------------

    auto baseKit = kit;

    ////

    GpuInitApiImpl gpuInitApi(baseKit);
    GpuExecApiImpl gpuExecApi(baseKit);

    ////

    const GpuProperties* gpuProperties = &gpuPropertiesInternal;
    const GpuContext* gpuContext = &gpuContextInternal;
    const GpuStream* gpuStream = &gpuStreamInternal;

    if (level == Level::InitExternal)
    {
        REQUIRE(args.externalContext);
        auto& externalContext = *args.externalContext;

        gpuProperties = &externalContext.gpuProperties;
        gpuContext = &externalContext.gpuContext;
        gpuStream = &externalContext.gpuStream;
    }

    ////

    auto kitEx = kitCombine
    (
        baseKit,
        GpuApiImplKit{gpuInitApi, gpuExecApi},
        GpuPropertiesKit{*gpuProperties},
        GpuCurrentContextKit{*gpuContext},
        GpuCurrentStreamKit{*gpuStream}
    );

    ////

    auto gpuShellTarget = GpuShellTarget::O | [&] (stdPars(GpuShellKit))
    {
        processWithGpu(args, stdPassThruKit(kitCombine(baseKit, kit)));
    };

    gpuShell.execCyclicShell(gpuShellTarget, stdPassKit(kitEx));
}

//================================================================
//
// MinimalShellImpl::processWithGpu
//
//================================================================

void MinimalShellImpl::processWithGpu(const ProcessArgs& args, stdPars(ProcessWithGpuKit))
{

    args.sysAllocHappened = false;

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
    // Engine module state memory.
    //
    //----------------------------------------------------------------

    MemoryUsage engineStateUsage;
    ReallocActivity engineStateActivity;

    {
        GpuProhibitedExecApiThunk prohibitedApi{kit};
        auto countKit = kitReplace(kit, prohibitedApi.getKit());
        auto execKit = kit;

        ////

        auto reallocValid = [&]
        {
            return args.engineModule.reallocValid();
        };

        auto realloc = [&] (stdPars(auto))
        {
            auto joinKit = kit.dataProcessing ? execKit : countKit;
            args.engineModule.realloc(stdPassThruKit(kitCombine(kit, joinKit)));
        };

        ////

        auto reallocTarget = memControllerReallocThunk(reallocValid, realloc);

        args.engineMemory.handleStateRealloc(reallocTarget, kit, engineStateUsage, engineStateActivity, stdPass);
    }

    ////

    REQUIRE(engineStateActivity.fastAllocCount <= 1);
    REQUIRE(engineStateActivity.sysAllocCount <= 1);

    ////

    if (engineStateActivity.sysAllocCount)
        args.sysAllocHappened = true;

    //----------------------------------------------------------------
    //
    // Count engine module temp memory.
    //
    //----------------------------------------------------------------

    MemoryUsage engineTempUsage;
    ReallocActivity engineTempActivity;

    {
        //
        // Pipe control on memory counting stage: advance 1 frame
        //

        PipeControl pipeControl{0, false};
        auto kitEx = kitCombine(kit, PipeControlKit(pipeControl));

        ////

        GpuProhibitedExecApiThunk prohibitedApi{kitEx}; // Prohibit GPU calls on counting.
        auto countKit = kitReplace(kitEx, prohibitedApi.getKit());

        ////

        auto processThunk = memControllerProcessThunk | [&] (stdPars(auto))
        {
            processWithAllocators(args, stdPassThruKit(kitCombine(countKit, kit)));
        };

        args.engineMemory.processCountTemp(processThunk, engineTempUsage, engineTempActivity, stdPassKit(kitEx));
    }

    //----------------------------------------------------------------
    //
    // Reallocate temp memory pools (if necessary).
    //
    //----------------------------------------------------------------

    args.engineMemory.handleTempRealloc(engineTempUsage, kit, engineTempActivity, stdPass);

    REQUIRE(engineTempActivity.fastAllocCount <= 1);
    REQUIRE(engineTempActivity.sysAllocCount <= 1);

    ////

    if (engineTempActivity.sysAllocCount)
        args.sysAllocHappened = true;

    ////

    if (displayMemoryUsage || uncommonActivity(engineStateActivity, engineTempActivity))
        memoryUsageReport(STR("Engine"), engineStateUsage, engineTempUsage, engineStateActivity, engineTempActivity, kit);

    //----------------------------------------------------------------
    //
    // Pure counting mode.
    //
    //----------------------------------------------------------------

    if_not (args.runExecutionPhase)
        return;

    //----------------------------------------------------------------
    //
    // Execute process with memory distribution.
    //
    //----------------------------------------------------------------

    {
        //
        // Pipe control on execution stage: advance 0 frames (rollback 1 frame)
        //

        PipeControl pipeControl{1, false};
        auto kitEx = kitCombine(kit, PipeControlKit(pipeControl));

        ////

        auto processTarget = memControllerProcessThunk | [&] (stdPars(auto))
        {
            processWithAllocators(args, stdPassThruKit(kitCombine(kit, kitEx)));
        };

        ////

        MemoryUsage actualEngineTempUsage;

        args.engineMemory.processAllocTemp(processTarget, kitEx, actualEngineTempUsage, stdPass);

        CHECK(actualEngineTempUsage == engineTempUsage);
    }
}

//================================================================
//
// MinimalShellImpl::processWithAllocators
//
//================================================================

void MinimalShellImpl::processWithAllocators(const ProcessArgs& args, stdPars(ProcessWithAllocatorsKit))
{
    stdScopedBegin;

    //----------------------------------------------------------------
    //
    // Headers.
    //
    //----------------------------------------------------------------

    bool alternative = displayParams.alternative();

    ////

    if (altVersionMonitor.active(5.f, kit))
        printMsgL(kit, STR("Alternative Version: %"), alternative, alternative ? msgWarn : msgInfo);

    //----------------------------------------------------------------
    //
    // Disabled kits.
    //
    //----------------------------------------------------------------

    SetBusyStatusNull setBusyStatus;

    auto& oldKit = kit;

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

        bmpConsole->setLockstepCounter(bmpCounter);

        if (kit.dataProcessing)
            bmpCounter++;
    }

    ////

    GpuBaseConsoleByCpuThunk gpuBaseConsoleCpuThunk{*baseConsole, *baseOverlay};

    auto gpuBaseConsoleCpuThunkEmpty = (baseConsole == &baseConsoleNull) && (baseOverlay == &baseOverlayNull);

    //----------------------------------------------------------------
    //
    // GPU base console.
    //
    //----------------------------------------------------------------

    GpuBaseConsole* gpuBaseConsole = &gpuBaseConsoleCpuThunk;

    GpuBaseConsoleSplitter gpuBaseConsoleSplitter{*kit.gpuBaseConsole, gpuBaseConsoleCpuThunk};

    ////

    if (kit.gpuBaseConsole)
    {
        if (gpuBaseConsoleCpuThunkEmpty)
            gpuBaseConsole = kit.gpuBaseConsole;
        else
            gpuBaseConsole = &gpuBaseConsoleSplitter;
    }

    //----------------------------------------------------------------
    //
    // GPU image console.
    //
    //----------------------------------------------------------------

    using namespace gpuImageConsoleImpl;

    ////

    GpuBaseConsoleProhibitThunk gpuBaseConsoleProhibited;

    if_not (kit.verbosity >= Verbosity::On)
        gpuBaseConsole = &gpuBaseConsoleProhibited;

    ////

    GpuImageConsoleThunk gpuImageConsole(*gpuBaseConsole, displayParams.displayMode(), displayParams.vectorMode(), kit);

    GpuImageConsoleKit gpuImageConsoleKit{gpuImageConsole};

    //----------------------------------------------------------------
    //
    // Display params and kit.
    //
    //----------------------------------------------------------------

    DisplayParamsThunk displayParamsThunk{point(1), kit.desiredOutputSize, displayParams}; // Screen size not supported well.

    ////

    auto kitEx = kitCombine
    (
        kit,
        gpuImageConsoleKit,
        displayParamsThunk.getKit()
    );

    ////

    args.engineModule.process(stdPassKit(kitEx));

    ////

    stdScopedEnd;
}

//----------------------------------------------------------------

}
