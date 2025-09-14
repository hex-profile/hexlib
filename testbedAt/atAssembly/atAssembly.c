#include "atAssembly.h"

#include <algorithm>

#include "atAssembly/frameAdvanceKit.h"
#include "atAssembly/frameChange.h"
#include "testModule/testModule.h"
#include "cfgTools/boolSwitch.h"
#include "compileTools/classContext.h"
#include "cfgTools/cfgSimpleString.h"
#include "cfgVars/configFile/configFile.h"
#include "errorLog/debugBreak.h"
#include "formattedOutput/errorBreakThunks.h"
#include "formattedOutput/userOutputThunks.h"
#include "gpuImageConsoleImpl/gpuImageConsoleImpl.h"
#include "gpuLayer/gpuCallsProhibition.h"
#include "gpuShell/gpuShell.h"
#include "interfaces/fileTools.h"
#include "memController/memController.h"
#include "memController/memoryUsageReport.h"
#include "overlayTakeover/overlayTakeoverThunk.h"
#include "profilerShell/profilerShell.h"
#include "signalsTools/legacySignalsImpl.h"
#include "storage/classThunks.h"
#include "storage/optionalObject.h"
#include "storage/rememberCleanup.h"
#include "tests/testShell/testShell.h"
#include "timerImpl/timerImpl.h"
#include "userOutput/paramMsg.h"
#include "userOutput/printMsgEx.h"
#include "videoPreprocessor/videoPreprocessor.h"
#include "imageRead/positionTools.h"
#include "kits/userPoint.h"

namespace atStartup {

//================================================================
//
// ProcessFinalKit
//
//================================================================

using ProcessProfilerKit = KitCombine<ProcessKit, TimerKit, OverlayTakeoverKit, UserPointKit, FrameAdvanceKit, ProfilerKit>;
using ProcessFinalKit = KitCombine<ProcessProfilerKit, gpuShell::GpuShellKit>;

using TargetReallocKit = ProcessFinalKit;
using TargetProcessKit = KitCombine<ProcessFinalKit, PipeControlKit>;

//================================================================
//
// ToolModule
//
//================================================================

using ToolModule = videoPreprocessor::VideoPreprocessor;
using ToolTarget = videoPreprocessor::VideoPrepTarget;
using ToolTargetProcessKit = videoPreprocessor::ProcessTargetKit;

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

    void realloc(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseKit);
        TargetReallocKit joinKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

        toolModule.realloc(stdPassThruKit(kitCombine(kit, joinKit)));
    }

public:

    inline ToolModuleReallocThunk(ToolModule& toolModule, const TargetReallocKit& baseKit)
        : toolModule(toolModule), baseKit(baseKit) {}

private:

    ToolModule& toolModule;
    TargetReallocKit baseKit;

};

//================================================================
//
// ToolModuleProcessThunk
//
//================================================================

class ToolModuleProcessThunk : public MemControllerProcessTarget
{

public:

    void process(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseKit);
        TargetProcessKit joinKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

        toolModule.process(toolTarget, stdPassThruKit(kitCombine(kit, joinKit, VerbosityKit(Verbosity::On))));
    }

public:

    inline ToolModuleProcessThunk(ToolModule& toolModule, ToolTarget& toolTarget, const TargetProcessKit& baseKit)
        : toolModule(toolModule), toolTarget(toolTarget), baseKit(baseKit) {}

private:

    ToolModule& toolModule;
    ToolTarget& toolTarget;
    TargetProcessKit baseKit;

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

using EngineBaseKit = KitCombine<ErrorLogKit, MsgLogExKit, MsgLogsKit, TimerKit, OverlayTakeoverKit, ProfilerKit, gpuShell::GpuShellKit>;

//================================================================
//
// EngineReallocThunk
//
//================================================================

class EngineReallocThunk : public MemControllerReallocTarget
{

public:

    bool reallocValid() const {return engine.reallocValid();}

    void realloc(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseGpuKit);
        EngineBaseKit gpuKit = kit.dataProcessing ? baseGpuKit : kitReplace(baseGpuKit, prohibitedApi.getKit());

        engine.realloc(stdPassThruKit(kitCombine(kit, gpuKit)));
    }

public:

    inline EngineReallocThunk(TestModule& engine, const EngineBaseKit& baseGpuKit)
        : engine(engine), baseGpuKit(baseGpuKit) {}

private:

    TestModule& engine;
    EngineBaseKit baseGpuKit;

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

    void process(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseKit);
        EngineBaseKit gpuKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

        auto resultKit = kitCombine(kit, gpuKit, extraKit);
        engine.process(stdPassThruKit(resultKit));
    }

public:

    inline EngineMemControllerTarget(TestModule& engine, const EngineBaseKit& baseKit, const ExtraKit& extraKit)
        : engine(engine), baseKit(baseKit), extraKit(extraKit) {}

private:

    TestModule& engine;
    EngineBaseKit baseKit;
    ExtraKit extraKit;

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

    void process(stdPars(ToolTargetProcessKit))
    {
        EngineMemControllerTarget engineThunk(engine, baseKit, kit);
        memController.processCountTemp(engineThunk, tempUsage, tempActivity, stdPassKit(baseKit));
    }

public:

    inline EngineTempCountToolTarget(TestModule& engine, MemController& memController, MemoryUsage& tempUsage, ReallocActivity& tempActivity, const EngineBaseKit& baseKit)
        : engine(engine), memController(memController), tempUsage(tempUsage), tempActivity(tempActivity), baseKit(baseKit) {}

private:

    TestModule& engine;
    MemController& memController;
    MemoryUsage& tempUsage;
    ReallocActivity& tempActivity;
    EngineBaseKit baseKit;

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

    void process(stdPars(ToolTargetProcessKit))
    {
        EngineMemControllerTarget engineThunk(engine, baseKit, kit);

        memController.processAllocTemp(engineThunk, baseKit, tempUsage, stdPassKit(baseKit));
    }

public:

    inline EngineTempDistribToolTarget(TestModule& engine, MemController& memController, MemoryUsage& tempUsage, const EngineBaseKit& baseKit)
        : engine(engine), memController(memController), tempUsage(tempUsage), baseKit(baseKit) {}

private:

    TestModule& engine;
    MemController& memController;
    MemoryUsage& tempUsage;
    EngineBaseKit baseKit;

};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// InputMetadataHandler
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

struct FileProperties
{
    bool exists = false;
    FileTime changeTime = 0;
    FileSize fileSize = 0;
};

inline bool operator==(const FileProperties& a, const FileProperties& b)
{
    return
        (a.exists == b.exists) &&
        (a.changeTime == b.changeTime) &&
        (a.fileSize == b.fileSize);
}

//================================================================
//
// getFileProperties
//
//================================================================

template <typename Kit>
void getFileProperties(const CharType* filename, FileProperties& result, stdPars(Kit))
{
    result = FileProperties{};

    if_not (fileTools::isFile(filename))
        return;

    result.exists = true;
    REQUIRE(fileTools::getChangeTime(filename, result.changeTime));
    REQUIRE(fileTools::getFileSize(filename, result.fileSize));
}

//================================================================
//
// InputMetadataHandler
//
//================================================================

class InputMetadataHandler
{

public:

    using UpdateKit = KitCombine<DiagnosticKit, LocalLogKit>;

    void checkSteady(const CharArray& inputName, CfgSerialization& serialization, bool& steady, stdPars(UpdateKit));
    void reloadFileOnChange(const CharArray& inputName, CfgSerialization& serialization, stdPars(UpdateKit));
    void saveVariablesOnChange(CfgSerialization& serialization, stdPars(UpdateKit));

private:

    SimpleString currentInputName;
    SimpleString currentConfigName;
    FileProperties currentProperties;

};

//================================================================
//
// InputMetadataHandler::checkSteady
//
//================================================================

void InputMetadataHandler::checkSteady(const CharArray& inputName, CfgSerialization& serialization, bool& steady, stdPars(UpdateKit))
{
    steady = false;

    //
    // The same input name?
    //

    if_not (strEqual(inputName, currentInputName.str()))
        return;

    //
    // Config properties.
    //

    FileProperties properties;
    getFileProperties(currentConfigName.cstr(), properties, stdPass);

    if_not (properties == currentProperties)
        return;

    ////

    steady = true;
}

//================================================================
//
// InputMetadataHandler::reloadFileOnChange
//
//================================================================

void InputMetadataHandler::reloadFileOnChange(const CharArray& inputName, CfgSerialization& serialization, stdPars(UpdateKit))
{
    //----------------------------------------------------------------
    //
    // Check steadiness.
    //
    //----------------------------------------------------------------

    bool steady = false;

    checkSteady(inputName, serialization, steady, stdPass);

    if (steady)
        return;

    //----------------------------------------------------------------
    //
    // Remember to reset in case of error.
    //
    //----------------------------------------------------------------

    printMsgL(kit, STR("Reloading metadata config."), msgWarn);

    // cfgvarsResetValue(serialization);

    REMEMBER_CLEANUP_EX(resetStateCleanup, {resetObject(*this); cfgvarsResetValue(serialization);});

    //----------------------------------------------------------------
    //
    // Update all metadata.
    //
    //----------------------------------------------------------------

    currentInputName = inputName;
    REQUIRE(def(currentInputName));

    ////

    auto dot = STR(".");
    auto dotPos = std::find_end(inputName.ptr, inputName.ptr + inputName.size, dot.ptr, dot.ptr + dot.size);
    size_t usedLength = dotPos - inputName.ptr;

    currentConfigName.assign(inputName.ptr, usedLength);
    currentConfigName += ".json";
    REQUIRE(def(currentConfigName));

    ////

    getFileProperties(currentConfigName.cstr(), currentProperties, stdPass);

    //----------------------------------------------------------------
    //
    // Read metadata config file.
    //
    //----------------------------------------------------------------

    UniqueInstance<ConfigFile> metadataConfig;

    if (currentProperties.exists)
        metadataConfig->loadFile(currentConfigName, stdPass);

    //----------------------------------------------------------------
    //
    // Serialize vars.
    //
    //----------------------------------------------------------------

    metadataConfig->loadVars(serialization, true, stdPass);
    metadataConfig->saveVars(serialization, true, stdPass);

    errorBlock(metadataConfig->updateFile(true, stdPassNc)); // Correct the config file.

    // Update the file properties after correction.
    getFileProperties(currentConfigName.cstr(), currentProperties, stdPass);

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    resetStateCleanup.cancel();
}

//================================================================
//
// InputMetadataHandler::saveVariablesOnChange
//
//================================================================

void InputMetadataHandler::saveVariablesOnChange(CfgSerialization& serialization, stdPars(UpdateKit))
{
    //----------------------------------------------------------------
    //
    // Check steadiness.
    //
    //----------------------------------------------------------------

    if (cfgvarsSynced(serialization))
        return;

    if_not (currentProperties.exists)
        return;

    //----------------------------------------------------------------
    //
    // Update the config.
    //
    //----------------------------------------------------------------

    printMsg(kit.localLog, STR("Updating metadata config."), msgWarn);

    REMEMBER_CLEANUP_EX(resetStateCleanup, {resetObject(*this);});

    ////

    UniqueInstance<ConfigFile> metadataConfig;
    metadataConfig->loadFile(currentConfigName, stdPass);

    metadataConfig->saveVars(serialization, false, stdPass);
    metadataConfig->updateFile(false, stdPass);

    // Update the file properties.
    getFileProperties(currentConfigName.cstr(), currentProperties, stdPass);

    resetStateCleanup.cancel();
}

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
// AtEngineTestWrapper
//
//================================================================

class AtEngineTestWrapper : public TestModule
{

public:

    AtEngineTestWrapper(UniquePtr<TestModule> base)
        : base{move(base)} {}

    virtual void setInputResolution(const Point<Space>& frameSize)
        {base->setInputResolution(frameSize);}

    virtual void serialize(const ModuleSerializeKit& kit)
    {
        base->serialize(kit);
        testShell->serialize(kit);
    }

    virtual void inputMetadataSerialize(const InputMetadataSerializeKit& kit)
        {base->inputMetadataSerialize(kit);}

    virtual bool reallocValid() const
        {return base->reallocValid();}

    virtual void realloc(stdPars(ReallocKit))
        {base->realloc(stdPassThru);}

    virtual void dealloc()
        {base->dealloc();}

    virtual void inspectProcess(ProcessInspector& inspector)
        {return base->inspectProcess(inspector);}

    virtual void process(stdPars(ProcessKit))
    {
        auto processApi = testShell::Process::O | [&] (stdParsNull)
        {
            base->process(stdPass);
        };

        testShell->process(processApi, stdPassThru);
    }

private:

    UniquePtr<TestModule> base;
    UniqueInstance<TestShell> testShell;

};

//================================================================
//
// AtAssemblyImpl
//
//================================================================

class AtAssemblyImpl
{

public:

    void serialize(const CfgSerializeKit& kit);

    auto cfgSerialization()
    {
        return cfgSerializationThunk | [&] (auto& kit) {return serialize(kit);};
    }

public:

    void init(const TestModuleFactory& engineFactory, stdPars(InitKit));
    void finalize(stdPars(InitKit));
    void process(stdPars(ProcessKit));

public:

    void processWithProfiler(stdPars(ProcessProfilerKit));
    void processFinal(stdPars(ProcessFinalKit));

private:

    bool initialized = false;

    //
    // Signals support
    //

    ArrayMemory<int32> signalHist;

    //
    // Config file
    //

    static const CharType* getDefaultConfigEditor()
    {
        const CharType* result = getenv(CT("HEXLIB_TEXT_EDITOR"));
        if (!result) result = CT("notepad");
        return result;
    }

    UniqueInstance<ConfigFile> configFilePtr;
    ConfigFile& configFile = *configFilePtr;
    ConfigUpdateDecimator configUpdateDecimator;

    SimpleStringVar configEditor{getDefaultConfigEditor()};

    StandardSignal configEditSignal;

    //
    //
    //

    OverlayTakeoverID overlayOwnerID = OverlayTakeoverID::undefined();
    StandardSignal deactivateOverlay;

    BoolSwitch displayMemoryUsage{false};
    BoolSwitch debugBreakOnErrors{false};

    //
    // Frame change
    //

    FrameChangeDetector frameChangeDetector;

    //
    // Input metadata handler.
    //

    InputMetadataHandler inputMetadataHandler;

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
    UniquePtr<TestModule> engineModule;

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
        auto& oldKit = kit;
        ModuleSerializeKit kit = kitCombine(oldKit, OverlayTakeoverKit(overlayTakeover));

        //
        // AtAssembly
        //

        {
            CFG_NAMESPACE("~Shell");

            {
                CFG_NAMESPACE("Config");
                configEditor.serialize(kit, STR("Editor"));
                configEditSignal.serialize(kit, STR("Edit"), STR("`"), STR("Press Tilde"));
            }

            profilerShell.serialize(kit, true);

            gpuShell.serialize(kit, true);
            gpuContextHelper.serialize(kit);

            deactivateOverlay.serialize(kit, STR("Deactivate Overlay"), STR("\\"));
            displayMemoryUsage.serialize(kit, STR("Display Memory Usage"), STR("Ctrl+Shift+U"));

            debugBreakOnErrors.serialize(kit, STR("Debug Break On Errors"), STR("Ctrl+B"));

            toolModule.serialize(kit);

            {
                CFG_NAMESPACE("Tool Memory");
                toolMemory.serialize(kit);
            }

            {
                CFG_NAMESPACE("Engine Memory");
                engineMemory.serialize(kit);
            }
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

void AtAssemblyImpl::init(const TestModuleFactory& engineFactory, stdPars(InitKit))
{
    initialized = false;

    //
    // Create engine
    //

    engineModule = makeUnique<AtEngineTestWrapper>(engineFactory.create());

    REQUIRE(!!engineModule);

    //
    // Config file
    //

    SimpleString configFilename; configFilename << engineFactory.configName() << CT(".json");
    errorBlock(configFile.loadFile(configFilename, stdPassNc));
    REMEMBER_CLEANUP_EX(configFileCleanup, configFile.unloadFile());

    ////

    auto serialization = cfgSerializationThunk | [&] (auto& kit) {serialize(kit);};

    auto handleConfig = [&] ()
    {
        configFile.loadVars(serialization, true, stdPass);
        configFile.saveVars(serialization, true, stdPass);
        configFile.updateFile(true, stdPass);
    };

    errorBlock(handleConfig());

    //
    // Register signals
    //

    using namespace signalImpl;

    int32 signalCount{};
    registerSignals(serialization, kit.atSignalSet, signalCount, stdPass);

    signalHist.realloc(signalCount, cpuBaseByteAlignment, kit.malloc, stdPass);

    REMEMBER_CLEANUP_EX(signalsCleanup, {kit.atSignalSet.actsetClear(); signalHist.dealloc();});

    //
    // Profiler
    //

    profilerShell.init(stdPass);
    REMEMBER_CLEANUP_EX(profilerCleanup, profilerShell.deinit());

    //
    // GPU init
    //

    GpuInitApiImpl gpuInitApi(kit);
    gpuInitApi.initialize(stdPass);
    GpuInitKit gpuInitKit = gpuInitApi.getKit();

    ////

    gpuContextHelper.createContext(gpuProperties, gpuContext, stdPassKit(kitCombine(kit, gpuInitKit)));
    REMEMBER_CLEANUP_EX(gpuContextCleanup, gpuContext.clear());

    ////

    gpuInitKit.gpuStreamCreation.createStream(gpuContext, true, gpuStream, stdPass);

    //
    // Record success
    //

    signalsCleanup.cancel();
    configFileCleanup.cancel();
    profilerCleanup.cancel();
    gpuContextCleanup.cancel();

    initialized = true;
}

//================================================================
//
// AtAssemblyImpl::finalize
//
//================================================================

void AtAssemblyImpl::finalize(stdPars(InitKit))
{
    if_not (initialized)
        return;

    //
    // Make finalization work
    //

    auto serialization = cfgSerialization();

    errorBlock(configFile.saveVars(serialization, false, stdPassNc)) &&
    errorBlock(configFile.updateFile(false, stdPassNc));
}

//================================================================
//
// AtAssemblyImpl::processFinal
//
//================================================================

void AtAssemblyImpl::processFinal(stdPars(ProcessFinalKit))
{
    using namespace memController;

    Point<Space> inputFrameSize = kit.atVideoFrame.size();
    REQUIRE(!!engineModule);

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
    // Tool module state realloc (if needed).
    //
    //----------------------------------------------------------------

    MemoryUsage toolStateUsage;
    ReallocActivity toolStateActivity;

    {
        toolModule.setFrameSize(inputFrameSize);
        ToolModuleReallocThunk toolModuleThunk(toolModule, kit);
        toolMemory.handleStateRealloc(toolModuleThunk, kit, toolStateUsage, toolStateActivity, stdPass);
    }

    //----------------------------------------------------------------
    //
    // Set engine input parameters.
    //
    //----------------------------------------------------------------

    InputVideoNameKit inputVideoNameKit(kit.atVideoInfo.videofileName);

    ////

    auto metadataSerialization = cfgSerializationThunk | [&] (auto& kit)
    {
        engineModule->inputMetadataSerialize(kitCombine(kit, inputVideoNameKit));
    };

    ////

    Point<Space> engineFrameSize = toolModule.outputFrameSize();
    engineModule->setInputResolution(engineFrameSize);

    ////

    inputMetadataHandler.reloadFileOnChange(kit.atVideoInfo.videofileName, metadataSerialization, stdPass);
    REMEMBER_CLEANUP(errorBlock(inputMetadataHandler.saveVariablesOnChange(metadataSerialization, stdPassNc)));

    //----------------------------------------------------------------
    //
    // Engine module state realloc (if needed).
    //
    //----------------------------------------------------------------

    MemoryUsage engineStateUsage;
    ReallocActivity engineStateActivity;

    {
        EngineReallocThunk engineModuleThunk(*engineModule, kit);
        engineMemory.handleStateRealloc(engineModuleThunk, kit, engineStateUsage, engineStateActivity, stdPass);
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
    // First stage: Count temp memory.
    //
    //----------------------------------------------------------------

    MemoryUsage toolTempUsage;
    ReallocActivity toolTempActivity;

    MemoryUsage engineTempUsage;
    ReallocActivity engineTempActivity;

    {
        //
        // Pipeline control on counting stage:
        // if input frame is repeated, rollback 1 frame (advance 0 frames), else rollback 0 frames (advance 1 frame).
        //

        PipeControl pipeControl{frameRepetition, false};
        TargetProcessKit kitEx = kitCombine(kit, PipeControlKit(pipeControl));

        ////

        EngineTempCountToolTarget engineThunk(*engineModule, engineMemory, engineTempUsage, engineTempActivity, kit);
        ToolModuleProcessThunk toolModuleThunk(toolModule, engineThunk, kitEx);

        toolMemory.processCountTemp(toolModuleThunk, toolTempUsage, toolTempActivity, stdPass);
    }

    //----------------------------------------------------------------
    //
    // Reallocate tool temp memory (if necessary).
    //
    //----------------------------------------------------------------

    toolMemory.handleTempRealloc(toolTempUsage, kit, toolTempActivity, stdPass);

    REQUIRE(toolTempActivity.fastAllocCount <= 1);
    REQUIRE(toolTempActivity.sysAllocCount <= 1);
    REQUIRE(toolStateActivity.fastAllocCount <= 1);
    REQUIRE(toolStateActivity.sysAllocCount <= 1);

    if (displayMemoryUsage || uncommonActivity(toolStateActivity, toolTempActivity))
        memoryUsageReport(STR("Tool"), toolStateUsage, toolTempUsage, toolStateActivity, toolTempActivity, kit);

    //----------------------------------------------------------------
    //
    // Reallocate engine temp memory (if necessary).
    //
    //----------------------------------------------------------------

    engineMemory.handleTempRealloc(engineTempUsage, kit, engineTempActivity, stdPass);

    REQUIRE(engineTempActivity.fastAllocCount <= 1);
    REQUIRE(engineTempActivity.sysAllocCount <= 1);
    REQUIRE(engineStateActivity.fastAllocCount <= 1);
    REQUIRE(engineStateActivity.sysAllocCount <= 1);

    if (displayMemoryUsage || uncommonActivity(engineStateActivity, engineTempActivity))
        memoryUsageReport(STR("Engine"), engineStateUsage, engineTempUsage, engineStateActivity, engineTempActivity, kit);

    //----------------------------------------------------------------
    //
    // Last stage: Real data processing with temp memory distribution.
    //
    //----------------------------------------------------------------

    {
        //
        // Pipeline control on execution stage:
        // in any case, rollback 1 frame (stay on the same frame), because all neccessary advancing
        // was made on successful counting stage.
        //

        PipeControl pipeControl{1, false};
        TargetProcessKit kitEx = kitCombine(kit, PipeControlKit(pipeControl));

        ////

        MemoryUsage actualEngineTempUsage;
        EngineTempDistribToolTarget engineThunk(*engineModule, engineMemory, actualEngineTempUsage, kit);

        ToolModuleProcessThunk toolModuleThunk(toolModule, engineThunk, kitEx);

        MemoryUsage actualToolTempUsage;
        toolMemory.processAllocTemp(toolModuleThunk, kit, actualToolTempUsage, stdPass);

        CHECK(actualToolTempUsage == toolTempUsage);
        CHECK(actualEngineTempUsage == engineTempUsage);
    }
}

//================================================================
//
// AtAssemblyImpl::processWithProfiler
//
//================================================================

void AtAssemblyImpl::processWithProfiler(stdPars(ProcessProfilerKit))
{
    //----------------------------------------------------------------
    //
    // Edit config
    //
    //----------------------------------------------------------------

    auto serialization = cfgSerialization();

    if (configEditSignal)
    {
        auto action = [&] ()
        {
            // Flush current vars to disk
            configFile.saveVars(serialization, false, stdPass);

            // Edit the file
            configFile.editFile(configEditor(), stdPass);

            // Load vars
            configFile.loadVars(serialization, true, stdPass);

            // Fix incorrect values in the file
            configFile.saveVars(serialization, true, stdPass);
            configFile.updateFile(true, stdPass);
        };

        errorBlock(action());
    }

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

    GpuInitApiImpl gpuInitApi(kit);
    GpuExecApiImpl gpuExecApi(kit);

    ////

    auto gpuShellToolkit = kitCombine
    (
        kit,
        gpuShell::GpuApiImplKit(gpuInitApi, gpuExecApi),
        GpuPropertiesKit(gpuProperties),
        GpuCurrentContextKit(gpuContext),
        GpuCurrentStreamKit(gpuStream)
    );

    ////

    auto baseKit = kit;

    auto gpuExecToAssembly = gpuShell::GpuShellTarget::O | [&] (stdPars(gpuShell::GpuShellKit))
    {
        processFinal(stdPassThruKit(kitCombine(baseKit, kit)));
    };

    gpuShell.execCyclicShell(gpuExecToAssembly, stdPassKit(gpuShellToolkit));

    //----------------------------------------------------------------
    //
    // Update config (once per a sec if there are any modifications)
    //
    //----------------------------------------------------------------

    if (configUpdateDecimator.shouldUpdate(kit.timer))
    {
        bool updateHappened = false;

        errorBlock(configFile.saveVars(serialization, false, updateHappened, stdPassNc)) &&
        errorBlock(configFile.updateFile(false, stdPassNc));

        if (updateHappened)
            printMsgL(kit, STR("Saving config file."), msgWarn);
    }
}

//================================================================
//
// AtAssemblyImpl::process
//
//================================================================

void AtAssemblyImpl::process(stdPars(ProcessKit))
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

    MsgLogExBreakShell msgLogEx(kit.msgLogEx, debugBreakOnErrors);
    MsgLogExKit msgLogExKit(msgLogEx);

    ////

    auto& oldKit = kit;

    auto kit = kitReplace(oldKit, kitCombine(msgLogKit, errorLogKit, msgLogExKit));

    //----------------------------------------------------------------
    //
    // Handle signals.
    //
    //----------------------------------------------------------------

    auto serialization = cfgSerialization();

    ////

    using namespace signalImpl;

    SignalsOverview overview;
    prepareSignalHistogram(kit.atSignalTest, signalHist, overview);

    handleSignals(serialization, signalHist, overlayOwnerID, deactivateOverlay);

    //----------------------------------------------------------------
    //
    // Frame change detection
    //
    //----------------------------------------------------------------

    bool frameAdvance = false;
    frameChangeDetector.check(kit.atVideoInfo, frameAdvance, stdPass);

    //----------------------------------------------------------------
    //
    // Skip frame?
    //
    //----------------------------------------------------------------

    if_not (frameAdvance || overview.realEventsFound || !overview.anyEventsFound)
        return;

    kit.localLog.clear();
    kit.atImgConsole.clear(stdPass);

    //----------------------------------------------------------------
    //
    // Proceed to profiler shell
    //
    //----------------------------------------------------------------

    TimerImpl timer;

    ////

    PipeControl pipeControl{frameAdvance ? 0 : 1, false};

    UserPoint userPoint;

    userPoint.valid = kit.atUserPointValid;
    userPoint.floatPos = convertIndexToPos(kit.atUserPointIdx);
    userPoint.leftSet = !!overview.mouseLeftSet;
    userPoint.leftReset = !!overview.mouseLeftReset;
    userPoint.rightSet = !!overview.mouseRightSet;
    userPoint.rightReset = !!overview.mouseRightReset;

    ////

    OverlayTakeoverThunk overlayTakeover(overlayOwnerID);

    ////

    {
        auto& oldKit = kit;

        auto kit = kitCombine
        (
            oldKit,
            TimerKit(timer),
            OverlayTakeoverKit(overlayTakeover),
            PipeControlKit(pipeControl),
            UserPointKit(userPoint),
            FrameAdvanceKit(frameAdvance)
        );

        auto baseKit = kit;

        auto profilerTarget = ProfilerTarget::O | [&] (stdPars(ProfilerKit))
        {
            processWithProfiler(stdPassThruKit(kitCombine(kit, baseKit)));
        };

        profilerShell.process(profilerTarget, gpuProperties.totalThroughput, stdPass);
    }

    ////

    stdScopedEnd;
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

void AtAssembly::init(const TestModuleFactory& engineFactory, stdPars(InitKit))
    {instance->init(engineFactory, stdPassThru);}

void AtAssembly::finalize(stdPars(InitKit))
    {instance->finalize(stdPassThru);}

void AtAssembly::process(stdPars(ProcessKit))
    {instance->process(stdPassThru);}

//----------------------------------------------------------------

}
