#include "atAssembly.h"

#include "atAssembly/frameAdvanceKit.h"
#include "atAssembly/frameChange.h"
#include "atAssembly/toolModule/toolModule.h"
#include "atAssembly/videoPreprocessor/videoPreprocessor.h"
#include "cfgTools/boolSwitch.h"
#include "compileTools/classContext.h"
#include "configFile/cfgSimpleString.h"
#include "configFile/configFile.h"
#include "errorLog/debugBreak.h"
#include "fileToolsImpl/fileToolsImpl.h"
#include "formattedOutput/userOutputThunks.h"
#include "atEngine/atEngine.h"
#include "gpuImageVisualization/gpuImageConsoleImpl.h"
#include "gpuLayer/gpuCallsProhibition.h"
#include "gpuShell/gpuShell.h"
#include "memController/memController.h"
#include "memController/memoryUsageReport.h"
#include "overlayTakeover/overlayTakeoverThunk.h"
#include "profilerShell/profilerShell.h"
#include "signalsImpl/signalsImpl.h"
#include "storage/classThunks.h"
#include "storage/disposableObject.h"
#include "storage/rememberCleanup.h"
#include "timerImpl/timerImpl.h"
#include "userOutput/paramMsg.h"

namespace atStartup {

//================================================================
//
// ProcessFinalKit
//
//================================================================

KIT_COMBINE6(ProcessEnrichedKit, ProcessKit, TimerKit, OverlayTakeoverKit, UserPointKit, FileToolsKit, FrameAdvanceKit);
KIT_COMBINE2(ProcessProfilerKit, ProcessEnrichedKit, ProfilerKit);

KIT_COMBINE2(ProcessFinalKit, ProcessProfilerKit, gpuShell::GpuShellKit);

KIT_COMBINE1(ModuleReallocKit, ProcessFinalKit);
KIT_COMBINE2(ModuleProcessKit, ProcessFinalKit, PipeControlKit);

//================================================================
//
// uncommonActivity
//
//================================================================

inline bool uncommonActivity(const ReallocActivity& stateActivity, const ReallocActivity& tempActivity)
{
    return stateActivity.sysAllocCount || tempActivity.sysAllocCount || stateActivity.fastAllocCount;
}

//================================================================
//
// ToolModuleReallocThunk
//
//================================================================

class ToolModuleReallocThunk : public MemControllerReallocTarget
{

public:

    bool reallocValid() const
        {return toolModule.reallocValid();}

public:

    bool realloc(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseKit);
        ModuleReallocKit joinKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

        return toolModule.realloc(stdPassThruKit(kitCombine(kit, joinKit)));
    }

public:

    inline ToolModuleReallocThunk(ToolModule& toolModule, const ModuleReallocKit& baseKit)
        : toolModule(toolModule), baseKit(baseKit) {}

private:

    ToolModule& toolModule;
    ModuleReallocKit const baseKit;

};

//================================================================
//
// ToolModuleProcessThunk
//
//================================================================

class ToolModuleProcessThunk : public MemControllerProcessTarget
{

public:

    bool process(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseKit);
        ModuleProcessKit joinKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

        return toolModule.process(toolTarget, stdPassThruKit(kitCombine(kit, joinKit, OutputLevelKit(OUTPUT_ENABLED, 0))));
    }

public:

    inline ToolModuleProcessThunk(ToolModule& toolModule, ToolTarget& toolTarget, const ModuleProcessKit& baseKit)
        : toolModule(toolModule), toolTarget(toolTarget), baseKit(baseKit) {}

private:

    ToolModule& toolModule;
    ToolTarget& toolTarget;
    ModuleProcessKit const baseKit;

};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Engine
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// EngineBaseKit
//
//================================================================

KIT_COMBINE7(EngineBaseKit, ErrorLogKit, ErrorLogExKit, MsgLogsKit, TimerKit, OverlayTakeoverKit, ProfilerKit, gpuShell::GpuShellKit);

//================================================================
//
// EngineReallocThunk
//
//================================================================

class EngineReallocThunk : public MemControllerReallocTarget
{

public:

    bool reallocValid() const {return engine.reallocValid();}

    bool realloc(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseGpuKit);
        EngineBaseKit gpuKit = kit.dataProcessing ? baseGpuKit : kitReplace(baseGpuKit, prohibitedApi.getKit());

        return engine.realloc(stdPassThruKit(kitCombine(kit, gpuKit)));
    }

public:

    inline EngineReallocThunk(AtEngine& engine, const EngineBaseKit& baseGpuKit)
        : engine(engine), baseGpuKit(baseGpuKit) {}

private:

    AtEngine& engine;
    EngineBaseKit const baseGpuKit;

};

//================================================================
//
// EngineMemControllerTarget
//
//================================================================

class EngineMemControllerTarget : public MemControllerProcessTarget
{

    using ExtraKit = ToolTargetProcessKit;

public:

    bool reallocValid() const {return engine.reallocValid();}

    bool process(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseKit);
        EngineBaseKit gpuKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

        AtEngineProcessKit resultKit = kitCombine(kit, gpuKit, extraKit);
        return engine.process(stdPassThruKit(resultKit));
    }

public:

    inline EngineMemControllerTarget(AtEngine& engine, const EngineBaseKit& baseKit, const ExtraKit& extraKit)
        : engine(engine), baseKit(baseKit), extraKit(extraKit) {}

private:

    AtEngine& engine;
    EngineBaseKit const baseKit;
    ExtraKit const extraKit;

};

//================================================================
//
// EngineTempCountToolTarget
//
//================================================================

class EngineTempCountToolTarget : public ToolTarget
{

public:

    void inspectProcess(ProcessInspector& inspector) {engine.inspectProcess(inspector);}

public:

    bool process(stdPars(ToolTargetProcessKit))
    {
        stdBegin;

        EngineMemControllerTarget engineThunk(engine, baseKit, kit);

        MemoryUsage tempUsage;
        require(memController.processCountTemp(engineThunk, tempUsage, stdPassKit(baseKit)));

        maxTempUsage = maxOf(maxTempUsage, tempUsage);

        stdEnd;
    }

public:

    inline EngineTempCountToolTarget
    (
        AtEngine& engine,
        MemController& memController,
        MemoryUsage& maxTempUsage,
        EngineBaseKit const baseKit
    )
        :
        engine(engine), memController(memController), maxTempUsage(maxTempUsage), baseKit(baseKit)
    {
    }

private:

    AtEngine& engine;
    MemController& memController;
    MemoryUsage& maxTempUsage;
    EngineBaseKit const baseKit;

};

//================================================================
//
// EngineTempDistribToolTarget
//
//================================================================

class EngineTempDistribToolTarget : public ToolTarget
{

public:

    void inspectProcess(ProcessInspector& inspector) {engine.inspectProcess(inspector);}

public:

    bool process(stdPars(ToolTargetProcessKit))
    {
        stdBegin;

        EngineMemControllerTarget engineThunk(engine, baseKit, kit);

        MemoryUsage tempUsage;
        require(memController.processAllocTemp(engineThunk, baseKit, tempUsage, stdPassKit(baseKit)));

        maxTempUsage = maxOf(maxTempUsage, tempUsage);

        stdEnd;
    }

public:

    inline EngineTempDistribToolTarget
    (
        AtEngine& engine,
        MemController& memController,
        MemoryUsage& maxTempUsage,
        EngineBaseKit const baseKit
    )
        :
        engine(engine),
        memController(memController),
        maxTempUsage(maxTempUsage),
        baseKit(baseKit)
    {
    }

private:

    AtEngine& engine;
    MemController& memController;
    MemoryUsage& maxTempUsage;
    EngineBaseKit baseKit;

};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// AtAssembly
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// AtAssemblyImpl
//
//================================================================

class AtAssemblyImpl : public CfgSerialization
{

public:

    bool init(const AtEngineFactory& engineFactory, stdPars(InitKit));
    void finalize(stdPars(InitKit));
    bool process(stdPars(ProcessKit));
    void serialize(const CfgSerializeKit& kit);

public:

    bool processWithProfiler(stdPars(ProcessProfilerKit));
    bool processFinal(stdPars(ProcessFinalKit));

private:

    bool initialized = false;

    //
    // Signals support
    //

    int32 lastSignalCount = 0;
    ArrayMemory<int32> signalHist;

    //
    // Config file
    //

    ConfigFile configFile;
    ConfigUpdateDecimator configUpdateDecimator;
    SimpleStringVar configEditor;
    StandardSignal configEditSignal;

    uint32 overlayOwnerID = 0;
    StandardSignal deactivateOverlay;
    BoolSwitch<false> displayMemoryUsage;

    BoolSwitch<false> debugBreakOnErrors;

    //
    // Frame change
    //

    FrameChangeDetector frameChangeDetector;

    //
    // Profiler
    //

    ProfilerShell profilerShell;

    //
    // GPU layer
    //

    GpuContextHelper gpuContextHelper;
    GpuProperties gpuProperties;
    GpuContextOwner gpuContext;
    GpuStreamOwner gpuStream;

    gpuShell::GpuShellImpl gpuShell;

    //
    // Tool and engine
    //

    MemController toolMemory; // construct before the module, destruct after it
    ToolModule toolModule;

    MemController engineMemory; // construct before the module, destruct after it
    UniquePtr<AtEngine> engineModule;

};

//================================================================
//
// AtAssemblyImpl::serialize
//
//================================================================

void AtAssemblyImpl::serialize(const CfgSerializeKit& kit)
{
    {
        OverlayTakeoverThunk overlayTakeover(overlayOwnerID);
        const CfgSerializeKit& kitOld = kit;
        ModuleSerializeKit kit = kitCombine(kitOld, OverlayTakeoverKit(overlayTakeover, 0));

        //
        // AtAssembly
        //

        {
            CFG_NAMESPACE_MODULE("ZTestbed");

            {
                CFG_NAMESPACE_MODULE("Config");
                configEditSignal.serialize(kit, STR("Edit"), STR("`"), STR("Press Tilde"));
                kit.visitor(kit.scope, SerializeSimpleString(configEditor, STR("Editor")));
            }

            profilerShell.serialize(kit);
            gpuShell.serialize(kit);

            deactivateOverlay.serialize(kit, STR("Deactivate Overlay"), STR("\\"));
            displayMemoryUsage.serialize(kit, STR("Display Memory Usage"), STR("Ctrl+Shift+U"));

            debugBreakOnErrors.serialize(kit, STR("Debug Break On Errors"));

            toolModule.serialize(kit);
        }

        ////

        if (engineModule)
            engineModule->serialize(kit);
    }
}

//================================================================
//
// AtAssemblyImpl::init
//
//================================================================

bool AtAssemblyImpl::init(const AtEngineFactory& engineFactory, stdPars(InitKit))
{
    stdBegin;

    initialized = false;

    //
    // Create engine
    //

    engineModule = move(engineFactory.create());
    REQUIRE(!!engineModule);

    //
    //
    //

    FileToolsImpl fileTools;
    FileToolsKit fileToolsKit(fileTools, 0);

    //
    // Config file
    //

    CharType* defaultEditor = getenv(CT("HEXLIB_CONFIG_EDITOR"));
    if (defaultEditor == 0) defaultEditor = CT("notepad");

    configEditor = defaultEditor;
    REMEMBER_CLEANUP1_EX(configEditorCleanup, configEditor.clear(), SimpleString&, configEditor);

    configFile.loadFile(CT("testbed.cfg"), stdPassKit(kitCombine(kit, fileToolsKit)));
    REMEMBER_CLEANUP1_EX(configFileCleanup, configFile.unloadFile(), ConfigFile&, configFile);

    configFile.loadVars(*this);

    configFile.saveVars(*this, true);
    configFile.updateFile(true, stdPassKit(kitCombine(kit, fileToolsKit))); // fix potential cfg errors

    //
    // Register signals
    //

    using namespace signalImpl;
    registerSignals(*this, 0, kit.atSignalSet, lastSignalCount);
    REMEMBER_CLEANUP1_EX(signalsCleanup, kit.atSignalSet.actsetClear(), const InitKit&, kit);

    //
    // Profiler
    //

    require(profilerShell.init(stdPass));
    REMEMBER_CLEANUP1_EX(profilerCleanup, profilerShell.deinit(), ProfilerShell&, profilerShell);

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

    signalsCleanup.cancel();
    configEditorCleanup.cancel();
    configFileCleanup.cancel();
    profilerCleanup.cancel();
    gpuContextCleanup.cancel();

    initialized = true;

    stdEnd;
}

//================================================================
//
// AtAssemblyImpl::finalize
//
//================================================================

void AtAssemblyImpl::finalize(stdPars(InitKit))
{
    stdBegin;

    if_not (initialized)
        return;

    ////

    FileToolsImpl fileTools;
    FileToolsKit fileToolsKit(fileTools, 0);

    //
    // Make finalization work
    //

    configFile.saveVars(*this, false);
    configFile.updateFile(false, stdPassKit(kitCombine(kit, fileToolsKit)));

    ////

    stdEndv;
}

//================================================================
//
// AtAssemblyImpl::processFinal
//
//================================================================

bool AtAssemblyImpl::processFinal(stdPars(ProcessFinalKit))
{
    stdBegin;

    using namespace memController;

    Point<Space> inputFrameSize = kit.atVideoFrame.size();
    REQUIRE(!!engineModule);

    //----------------------------------------------------------------
    //
    // Give GPU control to the profiler
    //
    //----------------------------------------------------------------

    ProfilerDeviceKit profilerDeviceKit = kit;
    profilerShell.setDeviceControl(&profilerDeviceKit);
    REMEMBER_CLEANUP1(profilerShell.setDeviceControl(0), ProfilerShell&, profilerShell);

    //----------------------------------------------------------------
    //
    // Tool module state realloc (if needed)
    //
    //----------------------------------------------------------------

    MemoryUsage toolStateUsage;
    ReallocActivity toolStateActivity;

    {
        toolModule.setFrameSize(inputFrameSize);
        ToolModuleReallocThunk toolModuleThunk(toolModule, kit);
        require(toolMemory.handleStateRealloc(toolModuleThunk, kit, toolStateUsage, toolStateActivity, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Engine module state realloc (if needed)
    //
    //----------------------------------------------------------------

    Point<Space> engineFrameSize = toolModule.outputFrameSize();

    MemoryUsage engineStateUsage;
    ReallocActivity engineStateActivity;

    {
        engineModule->setFrameSize(engineFrameSize);
        EngineReallocThunk engineModuleThunk(*engineModule, kit);
        require(engineMemory.handleStateRealloc(engineModuleThunk, kit, engineStateUsage, engineStateActivity, stdPass));
    }

    REQUIRE(engineStateActivity.fastAllocCount <= 1);
    REQUIRE(engineStateActivity.sysAllocCount <= 1);

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    bool frameRepetition = !kit.frameAdvance;

    //----------------------------------------------------------------
    //
    // First stage: Count temp memory
    //
    //----------------------------------------------------------------

    MemoryUsage toolTempUsage;
    MemoryUsage engineTempUsage;

    {
        //
        // Pipeline control on counting stage:
        // if input frame is repeated, rollback 1 frame (advance 0 frames), else rollback 0 frames (advance 1 frame).
        //

        PipeControl pipeControl(frameRepetition, false);
        ModuleProcessKit kitEx = kitCombine(kit, PipeControlKit(pipeControl, 0));

        ////

        EngineTempCountToolTarget engineThunk(*engineModule, engineMemory, engineTempUsage, kit);
        ToolModuleProcessThunk toolModuleThunk(toolModule, engineThunk, kitEx);

        require(toolMemory.processCountTemp(toolModuleThunk, toolTempUsage, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Reallocate tool temp memory (if necessary)
    //
    //----------------------------------------------------------------

    ReallocActivity toolTempActivity;
    require(toolMemory.handleTempRealloc(toolTempUsage, kit, toolTempActivity, stdPass));

    REQUIRE(toolTempActivity.fastAllocCount <= 1);
    REQUIRE(toolTempActivity.sysAllocCount <= 1);
    REQUIRE(toolStateActivity.fastAllocCount <= 1);
    REQUIRE(toolStateActivity.sysAllocCount <= 1);

    if (displayMemoryUsage || uncommonActivity(toolStateActivity, toolTempActivity))
        memoryUsageReport(STR("Tool"), toolStateUsage, toolTempUsage, toolStateActivity, toolTempActivity, stdPass);

    //----------------------------------------------------------------
    //
    // Reallocate engine temp memory (if necessary)
    //
    //----------------------------------------------------------------

    ReallocActivity engineTempActivity;
    require(engineMemory.handleTempRealloc(engineTempUsage, kit, engineTempActivity, stdPass));

    REQUIRE(engineTempActivity.fastAllocCount <= 1);
    REQUIRE(engineTempActivity.sysAllocCount <= 1);
    REQUIRE(engineStateActivity.fastAllocCount <= 1);
    REQUIRE(engineStateActivity.sysAllocCount <= 1);

    if (displayMemoryUsage || uncommonActivity(engineStateActivity, engineTempActivity))
        memoryUsageReport(STR("Engine"), engineStateUsage, engineTempUsage, engineStateActivity, engineTempActivity, stdPass);

    //----------------------------------------------------------------
    //
    // Last stage: Real data processing with temp memory distribution
    //
    //----------------------------------------------------------------

    {
        //
        // Pipeline control on execution stage:
        // in any case, rollback 1 frame (advance 0), because all neccessary advancing
        // was made on counting stage.
        //

        PipeControl pipeControl(1, false);
        ModuleProcessKit kitEx = kitCombine(kit, PipeControlKit(pipeControl, 0));

        ////

        MemoryUsage actualEngineTempUsage;
        EngineTempDistribToolTarget engineThunk(*engineModule, engineMemory, actualEngineTempUsage, kit);

        ToolModuleProcessThunk toolModuleThunk(toolModule, engineThunk, kitEx);

        MemoryUsage actualToolTempUsage;
        require(toolMemory.processAllocTemp(toolModuleThunk, kit, actualToolTempUsage, stdPass));

        CHECK(actualToolTempUsage == toolTempUsage);
        CHECK(actualEngineTempUsage == engineTempUsage);
    }

    stdEnd;
}

//================================================================
//
// GpuShellExecAppImpl
//
//================================================================

template <typename BaseKit>
class GpuShellExecAppImpl : public gpuShell::GpuShellTarget
{

public:

    bool exec(stdPars(gpuShell::GpuShellKit))
        {return base.processFinal(stdPassThruKit(kitCombine(baseKit, kit)));}

    inline GpuShellExecAppImpl(AtAssemblyImpl& base, const BaseKit& baseKit)
        : base(base), baseKit(baseKit) {}

private:

    AtAssemblyImpl& base;
    BaseKit const baseKit;

};

//================================================================
//
// profilerGpuFlush
//
//================================================================

template <typename Kit>
void profilerGpuFlush(void* context)
{
    const Kit& kit = * (const Kit*) context;
    TRACE_ROOT_STD;
    DEBUG_BREAK_CHECK(kit.gpuStreamWaiting.waitStream(kit.gpuCurrentStream, stdPassThru));
}

//================================================================
//
// AtAssemblyImpl::processWithProfiler
//
//================================================================

bool AtAssemblyImpl::processWithProfiler(stdPars(ProcessProfilerKit))
{
    stdBegin;

    //----------------------------------------------------------------
    //
    // Edit config
    //
    //----------------------------------------------------------------

    if (configEditSignal)
    {
        // Flush current vars to disk
        configFile.saveVars(*this, false);

        // Edit the file
        configFile.editFile(configEditor(), stdPass);

        // Load vars
        configFile.loadVars(*this);

        // Fix incorrect values in the file
        configFile.saveVars(*this, true);
        configFile.updateFile(true, stdPass);
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    GpuInitApiImpl gpuInitApi(kit);
    GpuExecApiImpl gpuExecApi(kit);

    ////

    gpuShell::ExecCyclicToolkit gpuShellToolkit = kitCombine
    (
        kit,
        gpuShell::GpuApiImplKit(gpuInitApi, gpuExecApi),
        GpuPropertiesKit(gpuProperties),
        GpuCurrentContextKit(gpuContext, 0),
        GpuCurrentStreamKit(gpuStream, 0)
    );

    ////

    GpuShellExecAppImpl<ProcessProfilerKit> gpuExecToAssembly(*this, kit);
    require(gpuShell.execCyclicShell(gpuExecToAssembly, stdPassKit(gpuShellToolkit)));

    //----------------------------------------------------------------
    //
    // Update config (once per 2 sec if there are any modifications)
    //
    //----------------------------------------------------------------

    if (configUpdateDecimator.shouldUpdate(kit.timer))
    {
        configFile.saveVars(*this, false);
        configFile.updateFile(false, stdPass);
    }

    ////

    stdEnd;
}

//================================================================
//
// ProfilerTargetToAssembly
//
//================================================================

template <typename BaseKit>
class ProfilerTargetToAssembly : public ProfilerTarget
{

public:

    bool process(stdPars(ProfilerKit))
        {return base.processWithProfiler(stdPassThruKit(kitCombine(kit, baseKit)));}

    inline ProfilerTargetToAssembly(AtAssemblyImpl& base, const BaseKit& baseKit)
        : base(base), baseKit(baseKit) {}

private:

    AtAssemblyImpl& base;
    BaseKit baseKit;

};

//================================================================
//
// MsgLogBreakShell
//
//================================================================

class MsgLogBreakShell : public MsgLog
{

public:

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind) override
    {
        bool ok = base.addMsg(v, msgKind);

        if (msgKind == msgErr && debugBreakOnErrors)
            DEBUG_BREAK_INLINE();

        return ok;
    }

    bool clear() override
        {return base.clear();}

    bool update() override
        {return base.update();}
    
    bool isThreadProtected() const override
        {return base.isThreadProtected();}

    void lock() override
        {return base.lock();}

    void unlock() override
        {return base.unlock();}

private:
    
    CLASS_CONTEXT(MsgLogBreakShell, ((MsgLog&, base)) ((bool, debugBreakOnErrors)));

};

//================================================================
//
// ErrorLogBreakShell
//
//================================================================

class ErrorLogBreakShell : public ErrorLog
{

public:

    bool isThreadProtected() const override
    {
        return base.isThreadProtected();
    }

    void addErrorSimple(const CharType* message) override
    {
        base.addErrorSimple(message);

        if (debugBreakOnErrors)
            DEBUG_BREAK_INLINE();
    }

    void addErrorTrace(const CharType* message, TRACE_PARAMS(trace)) override
    {
        base.addErrorTrace(message, TRACE_PASSTHRU(trace));

        if (debugBreakOnErrors)
            DEBUG_BREAK_INLINE();
    }

private:
    
    CLASS_CONTEXT(ErrorLogBreakShell, ((ErrorLog&, base)) ((bool, debugBreakOnErrors)));

};

//================================================================
//
// ErrorLogExBreakShell
//
//================================================================

class ErrorLogExBreakShell : public ErrorLogEx
{

public:

    bool isThreadProtected() const override
    {
        return base.isThreadProtected();
    }

    bool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars)
    {
        bool ok = base.addMsgTrace(v, msgKind, stdNullPassThru);

        if (msgKind == msgErr && debugBreakOnErrors)
            DEBUG_BREAK_INLINE();

        return ok;
    }

private:
    
    CLASS_CONTEXT(ErrorLogExBreakShell, ((ErrorLogEx&, base)) ((bool, debugBreakOnErrors)));

};

//================================================================
//
// AtAssemblyImpl::process
//
//================================================================

bool AtAssemblyImpl::process(stdPars(ProcessKit))
{
    stdBeginScoped;

    REQUIRE_EX(initialized, printMsg(kit.localLog, STR("Initialization failed, processing is disabled"), msgWarn));

    //----------------------------------------------------------------
    //
    // Intercept error output and add debug break functionality.
    //
    //----------------------------------------------------------------

    MsgLogBreakShell msgLog(kit.msgLog, debugBreakOnErrors);
    MsgLogKit msgLogKit(msgLog, 0);

    ErrorLogBreakShell errorLog(kit.errorLog, debugBreakOnErrors);
    ErrorLogKit errorLogKit(errorLog, 0);

    ErrorLogExBreakShell errorLogEx(kit.errorLogEx, debugBreakOnErrors);
    ErrorLogExKit errorLogExKit(errorLogEx, 0);

    ////

    auto oldKit = kit;

    auto kit = kitReplace(oldKit, kitCombine(msgLogKit, errorLogKit, errorLogExKit));

    //----------------------------------------------------------------
    //
    // Signal histogram
    //
    //----------------------------------------------------------------

    using namespace signalImpl;

    if_not (signalHist.resize(lastSignalCount))
        CHECK(signalHist.realloc(lastSignalCount, cpuBaseByteAlignment, kit.malloc, stdPass));

    bool anyEventsFound = false;
    bool realEventsFound = false;
    bool mouseSignal = false;
    bool mouseSignalAlt = false;
    prepareSignalHistogram(kit.atSignalTest, signalHist, anyEventsFound, realEventsFound, mouseSignal, mouseSignalAlt);

    //----------------------------------------------------------------
    //
    // Feed signals
    //
    //----------------------------------------------------------------

    //
    // Feed the signals
    //

    uint32 prevOverlayID = overlayOwnerID;

    {
        FeedSignal visitor(signalHist);
        serialize(CfgSerializeKit(visitor, 0));
    }

    //
    // Deactivate overlay
    //

    OverlayTakeoverThunk overlayTakeover(overlayOwnerID);

    if (deactivateOverlay)
        overlayOwnerID = 0;

    //
    // If overlay owner has changed, re-feed all the signals
    // to clean outdated switches.
    //

    if (prevOverlayID != overlayOwnerID)
    {
        FeedSignal visitor(signalHist);
        serialize(CfgSerializeKit(visitor, 0));
    }

    //----------------------------------------------------------------
    //
    // Frame change detection
    //
    //----------------------------------------------------------------

    bool frameAdvance = false;

    if_not (frameChangeDetector.check(kit.atVideoInfo, frameAdvance, stdPass))
        frameAdvance = true;

    //----------------------------------------------------------------
    //
    // Skip frame?
    //
    //----------------------------------------------------------------

    if_not (frameAdvance || realEventsFound || !anyEventsFound)
        return true;

    kit.localLog.clear();
    kit.atImgConsole.clear(stdPass);

    //----------------------------------------------------------------
    //
    // Proceed to profiler shell
    //
    //----------------------------------------------------------------

    TimerImpl timer;
    FileToolsImpl fileTools;

    ////

    PipeControl pipeControl(frameAdvance ? 0 : 1, false);
    UserPoint userPoint(kit.atUserPointValid, kit.atUserPoint, mouseSignal, mouseSignalAlt);

    ////

    {
        auto saveKit = kit;

        auto kit = kitCombine
        (
            saveKit,
            TimerKit(timer, 0),
            OverlayTakeoverKit(overlayTakeover, 0),
            PipeControlKit(pipeControl, 0),
            UserPointKit(userPoint, 0),
            FileToolsKit(fileTools, 0),
            FrameAdvanceKit(frameAdvance, 0)
        );

        ProfilerTargetToAssembly<ProcessEnrichedKit> profilerTarget(*this, kit);
        require(profilerShell.process(profilerTarget, gpuProperties.totalThroughput, stdPass));
    }

    ////

    stdEndScoped;
}

//================================================================
//
// Thunks
//
//================================================================

AtAssembly::AtAssembly() 
    {}

AtAssembly::~AtAssembly()
    {}

bool AtAssembly::init(const AtEngineFactory& engineFactory, stdPars(InitKit))
    {return instance->init(engineFactory, stdPassThru);}

void AtAssembly::finalize(stdPars(InitKit))
    {return instance->finalize(stdPassThru);}

bool AtAssembly::process(stdPars(ProcessKit))
    {return instance->process(stdPassThru);}

//----------------------------------------------------------------

}