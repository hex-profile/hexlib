#include "atAssembly.h"

#include <algorithm>

#include "atAssembly/frameAdvanceKit.h"
#include "atAssembly/frameChange.h"
#include "cfgTools/boolSwitch.h"
#include "compileTools/classContext.h"
#include "configFile/cfgSimpleString.h"
#include "configFile/configFile.h"
#include "errorLog/debugBreak.h"
#include "fileToolsImpl/fileToolsImpl.h"
#include "formattedOutput/userOutputThunks.h"
#include "atEngine/atEngine.h"
#include "gpuImageConsoleImpl/gpuImageConsoleImpl.h"
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
#include "userOutput/printMsgEx.h"
#include "videoPreprocessor/videoPreprocessor.h"
#include "tests/testShell/testShell.h"
#include "formattedOutput/errorBreakThunks.h"

namespace atStartup {

//================================================================
//
// ProcessFinalKit
//
//================================================================

KIT_COMBINE7(ProcessProfilerKit, ProcessKit, TimerKit, OverlayTakeoverKit, UserPointKit, FileToolsKit, FrameAdvanceKit, ProfilerKit);
KIT_COMBINE2(ProcessFinalKit, ProcessProfilerKit, gpuShell::GpuShellKit);

KIT_COMBINE1(TargetReallocKit, ProcessFinalKit);
KIT_COMBINE2(TargetProcessKit, ProcessFinalKit, PipeControlKit);

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

    stdbool realloc(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseKit);
        TargetReallocKit joinKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

        return toolModule.realloc(stdPassThruKit(kitCombine(kit, joinKit)));
    }

public:

    inline ToolModuleReallocThunk(ToolModule& toolModule, const TargetReallocKit& baseKit)
        : toolModule(toolModule), baseKit(baseKit) {}

private:

    ToolModule& toolModule;
    TargetReallocKit const baseKit;

};

//================================================================
//
// ToolModuleProcessThunk
//
//================================================================

class ToolModuleProcessThunk : public MemControllerProcessTarget
{

public:

    stdbool process(stdPars(memController::FastAllocToolkit))
    {
        GpuProhibitedExecApiThunk prohibitedApi(baseKit);
        TargetProcessKit joinKit = kit.dataProcessing ? baseKit : kitReplace(baseKit, prohibitedApi.getKit());

        return toolModule.process(toolTarget, stdPassThruKit(kitCombine(kit, joinKit, VerbosityKit(Verbosity::On))));
    }

public:

    inline ToolModuleProcessThunk(ToolModule& toolModule, ToolTarget& toolTarget, const TargetProcessKit& baseKit)
        : toolModule(toolModule), toolTarget(toolTarget), baseKit(baseKit) {}

private:

    ToolModule& toolModule;
    ToolTarget& toolTarget;
    TargetProcessKit const baseKit;

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

    stdbool realloc(stdPars(memController::FastAllocToolkit))
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

    stdbool process(stdPars(memController::FastAllocToolkit))
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

    stdbool process(stdPars(ToolTargetProcessKit))
    {
        EngineMemControllerTarget engineThunk(engine, baseKit, kit);

        MemoryUsage tempUsage;
        require(memController.processCountTemp(engineThunk, tempUsage, stdPassKit(baseKit)));

        maxTempUsage = maxOf(maxTempUsage, tempUsage);

        returnTrue;
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

    stdbool process(stdPars(ToolTargetProcessKit))
    {
        EngineMemControllerTarget engineThunk(engine, baseKit, kit);

        MemoryUsage tempUsage;
        require(memController.processAllocTemp(engineThunk, baseKit, tempUsage, stdPassKit(baseKit)));

        maxTempUsage = maxOf(maxTempUsage, tempUsage);

        returnTrue;
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
stdbool getFileProperties(const CharType* filename, FileProperties& result, stdPars(Kit))
{
    result = FileProperties{};

    if_not (kit.fileTools.fileExists(filename))
        returnTrue;

    result.exists = true;
    REQUIRE(kit.fileTools.getChangeTime(filename, result.changeTime));
    REQUIRE(kit.fileTools.getFileSize(filename, result.fileSize));

    returnTrue;
}

//================================================================
//
// InputMetadataHandler
//
//================================================================

class InputMetadataHandler
{

public:

    KIT_COMBINE3(UpdateKit, DiagnosticKit, LocalLogKit, FileToolsKit);

    stdbool checkSteady(const CharArray& inputName, CfgSerialization& serialization, bool& steady, stdPars(UpdateKit));
    stdbool reloadFileOnChange(const CharArray& inputName, CfgSerialization& serialization, stdPars(UpdateKit));
    stdbool saveVariablesOnChange(CfgSerialization& serialization, stdPars(UpdateKit));

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

stdbool InputMetadataHandler::checkSteady(const CharArray& inputName, CfgSerialization& serialization, bool& steady, stdPars(UpdateKit))
{
    steady = false;

    //
    // The same input name?
    //

    if_not (strEqual(inputName, currentInputName.charArray()))
        returnTrue;

    //
    // Config properties.
    //

    FileProperties properties;
    require(getFileProperties(currentConfigName.cstr(), properties, stdPass));
    
    if_not (properties == currentProperties)
        returnTrue;

    ////

    steady = true;

    returnTrue;
}

//================================================================
//
// InputMetadataHandler::reloadFileOnChange
//
//================================================================

stdbool InputMetadataHandler::reloadFileOnChange(const CharArray& inputName, CfgSerialization& serialization, stdPars(UpdateKit))
{
    //----------------------------------------------------------------
    //
    // Check steadiness.
    //
    //----------------------------------------------------------------

    bool steady = false;

    require(checkSteady(inputName, serialization, steady, stdPass));

    if (steady)
        returnTrue;

    //----------------------------------------------------------------
    //
    // Remember to reset in case of error.
    //
    //----------------------------------------------------------------

    printMsgL(kit, STR("Reloading metadata config."), msgWarn);

    // cfgvarResetValue(serialization);

    REMEMBER_CLEANUP_EX(resetStateCleanup, {resetObject(*this); cfgvarResetValue(serialization);});

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
    currentConfigName += ".cfg";
    REQUIRE(def(currentConfigName));

    ////

    require(getFileProperties(currentConfigName.cstr(), currentProperties, stdPass));

    //----------------------------------------------------------------
    //
    // Read metadata config file.
    //
    //----------------------------------------------------------------

    auto metadataConfig = ConfigFile::create();

    if (currentProperties.exists)
        require(metadataConfig->loadFile(currentConfigName, stdPass));

    //----------------------------------------------------------------
    //
    // Serialize vars.
    //
    //----------------------------------------------------------------

    metadataConfig->loadVars(serialization);

    metadataConfig->saveVars(serialization, true);
    errorBlock(metadataConfig->updateFile(true, stdPass)); // Correct the config file.

    // Update the file properties after correction.
    require(getFileProperties(currentConfigName.cstr(), currentProperties, stdPass)); 

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    resetStateCleanup.cancel();

    returnTrue;
}

//================================================================
//
// InputMetadataHandler::saveVariablesOnChange
//
//================================================================

stdbool InputMetadataHandler::saveVariablesOnChange(CfgSerialization& serialization, stdPars(UpdateKit))
{
    //----------------------------------------------------------------
    //
    // Check steadiness.
    //
    //----------------------------------------------------------------

    if_not (cfgvarChanged(serialization))
        returnTrue;

    if_not (currentProperties.exists)
        returnTrue;

    //----------------------------------------------------------------
    //
    // Update the config.
    //
    //----------------------------------------------------------------

    printMsg(kit.localLog, STR("Updating metadata config."), msgWarn);

    REMEMBER_CLEANUP_EX(resetStateCleanup, {resetObject(*this);});

    ////

    auto metadataConfig = ConfigFile::create();
    require(metadataConfig->loadFile(currentConfigName, stdPass));

    metadataConfig->saveVars(serialization, false);
    require(metadataConfig->updateFile(false, stdPass));

    // Update the file properties.
    require(getFileProperties(currentConfigName.cstr(), currentProperties, stdPass)); 

    resetStateCleanup.cancel();

    returnTrue;
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
// AtAssemblyImpl
//
//================================================================

class AtAssemblyImpl : public CfgSerialization
{

public:

    stdbool init(const AtEngineFactory& engineFactory, stdPars(InitKit));
    void finalize(stdPars(InitKit));
    stdbool process(stdPars(ProcessKit));
    void serialize(const CfgSerializeKit& kit);

public:

    stdbool processWithProfiler(stdPars(ProcessProfilerKit));
    stdbool processFinal(stdPars(ProcessFinalKit));

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
        const CharType* result = getenv(CT("HEXLIB_CONFIG_EDITOR"));
        if (!result) result = CT("notepad");
        return result;
    }

    UniquePtr<ConfigFile> configFilePtr = ConfigFile::create();
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
        ModuleSerializeKit kit = kitCombine(kitOld, OverlayTakeoverKit(overlayTakeover));

        //
        // AtAssembly
        //

        {
            CFG_NAMESPACE("~Shell");

            {
                CFG_NAMESPACE("Config");
                configEditSignal.serialize(kit, STR("Edit"), STR("`"), STR("Press Tilde"));
                configEditor.serialize(kit, STR("Editor"));
            }

            profilerShell.serialize(kit);
            
            gpuShell.serialize(kit);
            gpuContextHelper.serialize(kit);

            deactivateOverlay.serialize(kit, STR("Deactivate Overlay"), STR("\\"));
            displayMemoryUsage.serialize(kit, STR("Display Memory Usage"), STR("Ctrl+Shift+U"));

            debugBreakOnErrors.serialize(kit, STR("Debug Break On Errors"), STR("Ctrl+B"));

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

stdbool AtAssemblyImpl::init(const AtEngineFactory& engineFactory, stdPars(InitKit))
{
    initialized = false;

    //
    // Create engine
    //

    engineModule = testShell::testShellCreate(engineFactory.create());
    REQUIRE(!!engineModule);

    //
    //
    //

    FileToolsImpl fileTools;
    FileToolsKit fileToolsKit(fileTools);

    //
    // Config file
    //

    SimpleString configFilename; configFilename << engineModule->getName() << CT(".cfg");
    errorBlock(configFile.loadFile(configFilename, stdPassKit(kitCombine(kit, fileToolsKit))));
    REMEMBER_CLEANUP_EX(configFileCleanup, configFile.unloadFile());

    configFile.loadVars(*this);

    configFile.saveVars(*this, true);
    errorBlock(configFile.updateFile(true, stdPassKit(kitCombine(kit, fileToolsKit)))); // fix potential errors

    //
    // Register signals
    //

    using namespace signalImpl;

    int32 signalCount{};
    registerSignals(*this, 0, kit.atSignalSet, signalCount);

    require(signalHist.realloc(signalCount, cpuBaseByteAlignment, kit.malloc, stdPass));

    REMEMBER_CLEANUP_EX(signalsCleanup, {kit.atSignalSet.actsetClear(); signalHist.dealloc();});

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

    signalsCleanup.cancel();
    configFileCleanup.cancel();
    profilerCleanup.cancel();
    gpuContextCleanup.cancel();

    initialized = true;

    returnTrue;
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

    ////

    FileToolsImpl fileTools;
    FileToolsKit fileToolsKit(fileTools);

    //
    // Make finalization work
    //

    configFile.saveVars(*this, false);
    errorBlock(configFile.updateFile(false, stdPassKit(kitCombine(kit, fileToolsKit))));
}

//================================================================
//
// AtAssemblyImpl::processFinal
//
//================================================================

stdbool AtAssemblyImpl::processFinal(stdPars(ProcessFinalKit))
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
        require(toolMemory.handleStateRealloc(toolModuleThunk, kit, toolStateUsage, toolStateActivity, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Set engine input parameters.
    //
    //----------------------------------------------------------------

    class SerializeInputMetadata : public CfgSerialization
    {
        void serialize(const CfgSerializeKit& kit) 
            {engine.inputMetadataSerialize(kitCombine(kit, extraKit));}

        CLASS_CONTEXT(SerializeInputMetadata, ((AtEngine&, engine)) ((InputVideoNameKit, extraKit)));
    };

    InputVideoNameKit inputVideoNameKit(kit.atVideoInfo.videofileName);
    SerializeInputMetadata metadataSerialization(*engineModule, inputVideoNameKit);

    ////

    Point<Space> engineFrameSize = toolModule.outputFrameSize();
    engineModule->setInputResolution(engineFrameSize);

    ////

    require(inputMetadataHandler.reloadFileOnChange(kit.atVideoInfo.videofileName, metadataSerialization, stdPass));
    REMEMBER_CLEANUP(errorBlock(inputMetadataHandler.saveVariablesOnChange(metadataSerialization, stdPass)));

    //----------------------------------------------------------------
    //
    // Engine module state realloc (if needed).
    //
    //----------------------------------------------------------------

    MemoryUsage engineStateUsage;
    ReallocActivity engineStateActivity;

    {
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
    // First stage: Count temp memory.
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
        TargetProcessKit kitEx = kitCombine(kit, PipeControlKit(pipeControl));

        ////

        EngineTempCountToolTarget engineThunk(*engineModule, engineMemory, engineTempUsage, kit);
        ToolModuleProcessThunk toolModuleThunk(toolModule, engineThunk, kitEx);

        require(toolMemory.processCountTemp(toolModuleThunk, toolTempUsage, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Reallocate tool temp memory (if necessary).
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
    // Reallocate engine temp memory (if necessary).
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
    // Last stage: Real data processing with temp memory distribution.
    //
    //----------------------------------------------------------------

    {
        //
        // Pipeline control on execution stage:
        // in any case, rollback 1 frame (stay on the same frame), because all neccessary advancing
        // was made on successful counting stage.
        //

        PipeControl pipeControl(1, false);
        TargetProcessKit kitEx = kitCombine(kit, PipeControlKit(pipeControl));

        ////

        MemoryUsage actualEngineTempUsage;
        EngineTempDistribToolTarget engineThunk(*engineModule, engineMemory, actualEngineTempUsage, kit);

        ToolModuleProcessThunk toolModuleThunk(toolModule, engineThunk, kitEx);

        MemoryUsage actualToolTempUsage;
        require(toolMemory.processAllocTemp(toolModuleThunk, kit, actualToolTempUsage, stdPass));

        CHECK(actualToolTempUsage == toolTempUsage);
        CHECK(actualEngineTempUsage == engineTempUsage);
    }

    returnTrue;
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

    stdbool exec(stdPars(gpuShell::GpuShellKit))
        {return base.processFinal(stdPassThruKit(kitCombine(baseKit, kit)));}

    inline GpuShellExecAppImpl(AtAssemblyImpl& base, const BaseKit& baseKit)
        : base(base), baseKit(baseKit) {}

private:

    AtAssemblyImpl& base;
    BaseKit const baseKit;

};

//================================================================
//
// AtAssemblyImpl::processWithProfiler
//
//================================================================

stdbool AtAssemblyImpl::processWithProfiler(stdPars(ProcessProfilerKit))
{
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
        errorBlock(configFile.editFile(configEditor(), stdPass));

        // Load vars
        configFile.loadVars(*this);

        // Fix incorrect values in the file
        configFile.saveVars(*this, true);
        errorBlock(configFile.updateFile(true, stdPass));
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
        GpuCurrentContextKit(gpuContext),
        GpuCurrentStreamKit(gpuStream)
    );

    ////

    GpuShellExecAppImpl<ProcessProfilerKit> gpuExecToAssembly(*this, kit);
    require(gpuShell.execCyclicShell(gpuExecToAssembly, stdPassKit(gpuShellToolkit)));

    //----------------------------------------------------------------
    //
    // Update config (once per a sec if there are any modifications)
    //
    //----------------------------------------------------------------

    if (configUpdateDecimator.shouldUpdate(kit.timer))
    {
        bool updateHappened = false;
        configFile.saveVars(*this, false, &updateHappened);
        errorBlock(configFile.updateFile(false, stdPass));

        if (updateHappened)
            printMsgL(kit, STR("Saving config file"), msgWarn);
    }

    ////

    returnTrue;
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

    stdbool process(stdPars(ProfilerKit))
        {return base.processWithProfiler(stdPassThruKit(kitCombine(kit, baseKit)));}

    inline ProfilerTargetToAssembly(AtAssemblyImpl& base, const BaseKit& baseKit)
        : base(base), baseKit(baseKit) {}

private:

    AtAssemblyImpl& base;
    BaseKit baseKit;

};

//================================================================
//
// AtAssemblyImpl::process
//
//================================================================

stdbool AtAssemblyImpl::process(stdPars(ProcessKit))
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

    ////

    auto oldKit = kit;

    auto kit = kitReplace(oldKit, kitCombine(msgLogKit, errorLogKit, errorLogExKit));

    //----------------------------------------------------------------
    //
    // Handle signals.
    //
    //----------------------------------------------------------------

    using namespace signalImpl;

    SignalsOverview overview;
    prepareSignalHistogram(kit.atSignalTest, signalHist, overview);

    handleSignals(*this, signalHist, overlayOwnerID, deactivateOverlay);

    //----------------------------------------------------------------
    //
    // Frame change detection
    //
    //----------------------------------------------------------------

    bool frameAdvance = false;
    require(frameChangeDetector.check(kit.atVideoInfo, frameAdvance, stdPass));

    //----------------------------------------------------------------
    //
    // Skip frame?
    //
    //----------------------------------------------------------------

    if_not (frameAdvance || overview.realEventsFound || !overview.anyEventsFound)
        returnTrue;

    kit.localLog.clear();
    require(kit.atImgConsole.clear(stdPass));

    //----------------------------------------------------------------
    //
    // Proceed to profiler shell
    //
    //----------------------------------------------------------------

    TimerImpl timer;
    FileToolsImpl fileTools;

    ////

    PipeControl pipeControl(frameAdvance ? 0 : 1, false);
    UserPoint userPoint(kit.atUserPointValid, kit.atUserPoint, overview.mouseSignal, overview.mouseSignalAlt);

    ////

    OverlayTakeoverThunk overlayTakeover(overlayOwnerID);

    ////

    {
        auto oldKit = kit;

        auto kit = kitCombine
        (
            oldKit,
            TimerKit(timer),
            OverlayTakeoverKit(overlayTakeover),
            PipeControlKit(pipeControl),
            UserPointKit(userPoint),
            FileToolsKit(fileTools),
            FrameAdvanceKit(frameAdvance)
        );

        ProfilerTargetToAssembly<decltype(kit)> profilerTarget(*this, kit);
        require(profilerShell.process(profilerTarget, gpuProperties.totalThroughput, stdPass));
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

stdbool AtAssembly::init(const AtEngineFactory& engineFactory, stdPars(InitKit))
    {return instance->init(engineFactory, stdPassThru);}

void AtAssembly::finalize(stdPars(InitKit))
    {return instance->finalize(stdPassThru);}

stdbool AtAssembly::process(stdPars(ProcessKit))
    {return instance->process(stdPassThru);}

//----------------------------------------------------------------

}
