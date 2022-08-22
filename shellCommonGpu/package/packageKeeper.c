#include "packageKeeper.h"

#include <stdexcept>

#include "minimalShell/minimalShell.h"
#include "configFile/configFile.h"
#include "debugBridge/bridgeUsage/actionReceivingByBridge.h"
#include "debugBridge/bridgeUsage/actionSetupToBridge.h"
#include "debugBridge/bridgeUsage/baseImageConsoleToBridge.h"
#include "storage/rememberCleanup.h"
#include "dataAlloc/arrayMemory.h"
#include "signalsImpl/signalsImpl.h"
#include "userOutput/printMsgEx.h"
#include "overlayTakeover/overlayTakeoverThunk.h"

namespace packageImpl {
namespace packageKeeper {

//================================================================
//
// PackageKeeperImpl
//
//================================================================

class PackageKeeperImpl : public PackageKeeper
{

    //----------------------------------------------------------------
    //
    // Serialize.
    //
    //----------------------------------------------------------------

public:

    void serialize(const CfgSerializeKit& kit);

    Settings& settings() {return shell->settings();}

    //----------------------------------------------------------------
    //
    // Init and finalize helpers.
    //
    //----------------------------------------------------------------

public:

    stdbool init(const CharType* const configName, SerializeTarget& target, stdPars(StarterDebugKit));
    stdbool finalize(SerializeTarget& target, stdPars(StarterDebugKit));

    //----------------------------------------------------------------
    //
    // Process helper.
    //
    //----------------------------------------------------------------

public:

    stdbool process(ProcessTarget& target, bool warmup, stdPars(StarterDebugKit));

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

private:

    bool initialized = false;

    //----------------------------------------------------------------
    //
    // The engine memory & the minimal shell, the order of construction is important.
    //
    //----------------------------------------------------------------

    UniquePtr<MinimalShell> shell = MinimalShell::create();

    MemController engineMemory;

    //----------------------------------------------------------------
    //
    // Signals.
    //
    //----------------------------------------------------------------

    ArrayMemory<int32> signalHist;
    BaseMousePos signalMousePos;

    OverlayTakeoverID overlayOwnerID = OverlayTakeoverID::undefined();
    StandardSignal deactivateOverlay;

    //----------------------------------------------------------------
    //
    // Config file.
    //
    //----------------------------------------------------------------

    bool configFileActive = false;
    UniquePtr<ConfigFile> configFilePtr = ConfigFile::create();
    ConfigFile& configFile = *configFilePtr;

};

//----------------------------------------------------------------

UniquePtr<PackageKeeper> PackageKeeper::create()
{
    return makeUnique<PackageKeeperImpl>();
}

//================================================================
//
// PackageKeeperImpl::serialize
//
//================================================================

void PackageKeeperImpl::serialize(const CfgSerializeKit& kit)
{
    shell->serialize(kit);

    ////

    {
        CFG_NAMESPACE("~Shell");
        deactivateOverlay.serialize(kit, STR("Deactivate Overlay"), STR("\\"));
    }
}

//================================================================
//
// PackageKeeperImpl::init
//
//================================================================

stdbool PackageKeeperImpl::init(const CharType* const configName, SerializeTarget& target, stdPars(StarterDebugKit))
{

    REQUIRE(!initialized);

    //----------------------------------------------------------------
    //
    // Config file.
    //
    //----------------------------------------------------------------

    auto serialization = overlaySerializationThunk(target, overlayOwnerID);

    ////

    configFileActive = configName && *configName;

    ////

    if (configFileActive)
    {
        errorBlock(configFile.loadFile(SimpleString{configName}, stdPass));
        configFile.loadVars(serialization);
        configFile.saveVars(serialization, true);
        errorBlock(configFile.updateFile(true, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Register signals.
    //
    //----------------------------------------------------------------

    auto bridgeActionSetup = kit.debugBridge.actionSetup();
    REQUIRE(bridgeActionSetup);

    BaseActionSetupToBridge actionSetup{*bridgeActionSetup, stdPass};

    ////

    REMEMBER_CLEANUP_EX(signalsCleanup, {actionSetup.actsetClear(); signalHist.dealloc(); signalMousePos = BaseMousePos{};});

    ////

    require(actionSetup.actsetClear());

    ////

    int32 signalCount{};
    signalImpl::registerSignals(serialization, 0, actionSetup, signalCount);

    ////

    require(actionSetup.actsetUpdate());

    ////

    require(signalHist.realloc(signalCount, cpuBaseByteAlignment, kit.malloc, stdPass));

    //----------------------------------------------------------------
    //
    // Shell init.
    //
    //----------------------------------------------------------------

    require(shell->init(stdPass));
  
    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    initialized = true;

    signalsCleanup.cancel();

    returnTrue;
}

//================================================================
//
// PackageKeeperImpl::finalize
//
//================================================================

stdbool PackageKeeperImpl::finalize(SerializeTarget& target, stdPars(StarterDebugKit))
{
    REQUIRE(initialized);

    //----------------------------------------------------------------
    //
    // Save config if active.
    //
    //----------------------------------------------------------------

    auto serialization = overlaySerializationThunk(target, overlayOwnerID);

    if (configFileActive)
    {
        configFile.saveVars(serialization, true);
        errorBlock(configFile.updateFile(false, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Make profiler report if profiling is active.
    //
    //----------------------------------------------------------------

    if (shell->profilingActive())
        errorBlock(shell->profilingReport(stdPass));

    returnTrue;
}

//================================================================
//
// WarmupKitDisabler
//
//================================================================

class WarmupKitDisabler : public ProcessTarget
{

public:

    WarmupKitDisabler(ProcessTarget& base, bool warmup)
        : base{base}, warmup{warmup} {}

public:

    virtual void serialize(const ModuleSerializeKit& kit)
        {return base.serialize(kit);}

    virtual bool reallocValid() const
        {return base.reallocValid();}

    virtual stdbool realloc(stdPars(EngineReallocKit))
        {return base.realloc(stdPassThru);}

public:

    virtual stdbool process(stdPars(EngineProcessKit))
    {
        auto modifiedKits = kitCombine
        (
            ProfilerKit(!warmup ? kit.profiler : nullptr),
            VerbosityKit(!warmup ? kit.verbosity : Verbosity::Off)
        );

        return base.process(stdPassThruKit(kitReplace(kit, modifiedKits)));
    }

private:

    ProcessTarget& base;
    bool const warmup;

};

//================================================================
//
// PackageKeeperImpl::process
//
//================================================================

stdbool PackageKeeperImpl::process(ProcessTarget& target, bool warmup, stdPars(StarterDebugKit))
{
    REQUIRE(initialized);

    auto serialization = overlaySerializationThunk(target, overlayOwnerID);

    //----------------------------------------------------------------
    //
    // Handle signals.
    //
    //----------------------------------------------------------------

    auto bridgeReceiving = kit.debugBridge.actionReceiving();
    REQUIRE(bridgeReceiving);
    BaseActionReceivingByBridge actionReceiving{*bridgeReceiving, stdPass};

    ////

    using namespace signalImpl;

    SignalsOverview overview;
    prepareSignalHistogram(actionReceiving, signalHist, overview);

    ////

    handleSignals(serialization, signalHist, overlayOwnerID, deactivateOverlay);

    //----------------------------------------------------------------
    //
    // Debug bridge: Resetup actions.
    //
    //----------------------------------------------------------------

    if (overview.resetupActions)
    {
        auto bridgeActionSetup = kit.debugBridge.actionSetup();
        REQUIRE(bridgeActionSetup);

        BaseActionSetupToBridge actionSetup{*bridgeActionSetup, stdPass};

        ////

        REMEMBER_CLEANUP_EX(signalsCleanup, {actionSetup.actsetClear(); signalHist.dealloc();});

        ////

        printMsgL(kit, STR("Actions resetup"), msgWarn);

        ////

        require(actionSetup.actsetClear());

        ////

        int32 signalCount{};
        signalImpl::registerSignals(serialization, 0, actionSetup, signalCount);

        ////

        require(actionSetup.actsetUpdate());

        ////

        require(signalHist.realloc(signalCount, cpuBaseByteAlignment, kit.malloc, stdPass));

        ////

        signalsCleanup.cancel();
    }

    //----------------------------------------------------------------
    //
    // Debug bridge: Config operations.
    //
    //----------------------------------------------------------------

    {
        namespace db = debugBridge;
        db::ConfigSupport* configSupport = kit.debugBridge.configSupport();
        REQUIRE(configSupport);

        //
        // Config loader. Also updates file on disk if any.
        //

        auto configLoaderLambda = [&] (db::ArrayRef<const db::Char> config)
        {
            if_not (errorBlock(configFile.loadFromString(charArray(config.ptr, config.size), stdPass)))
                throw std::runtime_error("Config support error"); // Debug bridge uses exceptions.

            configFile.loadVars(serialization);

            errorBlock(configFile.updateFile(true, stdPass));
        };

        ////

        auto configLoader = db::configReceiverByLambda(configLoaderLambda);

        //
        // Update config vars. Also saves to disk if dirty.
        //

        auto configUpdateVars = [&] (stdPars(StarterDebugKit))
        {
            configFile.saveVars(serialization, false);
            errorBlock(configFile.updateFile(false, stdPass));
        };

        //
        // Save config operation.
        //

        if (overview.saveConfig)
        {
            auto configSaverLambda = [&] (const CharArray& str, stdNullPars)
            {
                require(blockExceptionsVoid(configSupport->saveConfig({str.ptr, str.size})));
                returnTrue;
            };

            auto configSaver = cfgVarsImpl::stringReceiverByLambda(configSaverLambda);

            ////

            configUpdateVars(stdPass);

            require(configFile.saveToString(configSaver, stdPass));
        }

        //
        // Load config operation.
        //

        if (overview.loadConfig)
        {
            require(blockExceptionsVoid(configSupport->loadConfig(configLoader)));
        }

        //
        // Edit config operation.
        //

        if (overview.editConfig)
        {
            auto configEditorLambda = [&] (const CharArray& str, stdNullPars)
            {
                require(blockExceptionsVoid(configSupport->editConfig({str.ptr, str.size}, configLoader)));
                returnTrue;
            };

            auto configEditor = cfgVarsImpl::stringReceiverByLambda(configEditorLambda);

            ////

            configUpdateVars(stdPass);

            require(configFile.saveToString(configEditor, stdPass));
        }
    }

    //----------------------------------------------------------------
    //
    // Debug bridge: Video overlay and user point.
    //
    //----------------------------------------------------------------

    auto* videoOverlay = kit.debugBridge.videoOverlay();
    REQUIRE(videoOverlay);

    ////

    if (deactivateOverlay)
        require(blockExceptionsVoid(videoOverlay->clear()));

    ////

    UserPoint userPoint;

    if (overview.mousePos.valid())
        signalMousePos = overview.mousePos;

    userPoint.valid = signalMousePos.valid();
    userPoint.position = signalMousePos.pos();

    userPoint.leftSet = (overview.mouseLeftSet != 0);
    userPoint.rightSet = (overview.mouseRightSet != 0);

    userPoint.leftReset = (overview.mouseLeftReset != 0);
    userPoint.rightReset = (overview.mouseRightReset != 0);

    ////

    BaseVideoOverlayToBridge baseOverlay{*kit.debugBridge.videoOverlay(), kit};

    ////

    auto kitEx = kitCombine
    (
        kit,
        minimalShell::BaseImageConsolesKit{nullptr, &baseOverlay},
        UserPointKit{userPoint}
    );

    //----------------------------------------------------------------
    //
    // Lockstep for debug images.
    //
    //----------------------------------------------------------------

    shell->settings().setImageSavingLockstepCounter(kit.dumpParams.dumpIndex);

    //----------------------------------------------------------------
    //
    // Call the target.
    //
    //----------------------------------------------------------------

    bool sysAllocHappened = false;

    WarmupKitDisabler targetEx{target, warmup};

    require(shell->process(targetEx, engineMemory, true, sysAllocHappened, stdPassKit(kitEx)));

    if (sysAllocHappened && !warmup)
        printMsgL(kit, STR("WARNING: System realloc."), msgWarn);

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
}
