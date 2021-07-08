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
    // Engine memory & minimal shell, construction order is important.
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

    OverlayTakeover::ID overlayOwnerID = 0;
    StandardSignal deactivateOverlay;

    //----------------------------------------------------------------
    //
    // Config file.
    //
    //----------------------------------------------------------------

    bool configActive = false;
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

    configActive = configName && *configName;

    ////

    if (configActive)
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

    REMEMBER_CLEANUP_EX(signalsCleanup, {actionSetup.actsetClear(); signalHist.dealloc();});

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

    if (configActive)
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

        ////

        auto configLoaderLambda = [&] (db::ArrayRef<const db::Char> config)
        {
            if_not (errorBlock(configFile.loadFromString(charArray(config.ptr, config.size), stdPass)))
                throw std::runtime_error("Config support error"); // Debug bridge uses exceptions.

            configFile.loadVars(serialization);
        };

        ////

        auto configLoader = db::configReceiverByLambda(configLoaderLambda);

        ////

        if (overview.saveConfig)
        {
            auto configSaverLambda = [&] (const CharArray& str, stdNullPars)
            {
                require(blockExceptionsVoid(configSupport->saveConfig({str.ptr, str.size})));
                returnTrue;
            };

            auto configSaver = cfgVarsImpl::stringReceiverByLambda(configSaverLambda);

            ////

            configFile.saveVars(serialization, false);

            require(configFile.saveToString(configSaver, stdPass));
        }

        ////

        if (overview.loadConfig)
        {
            require(blockExceptionsVoid(configSupport->loadConfig(configLoader)));
        }

        ////

        if (overview.editConfig)
        {
            auto configEditorLambda = [&] (const CharArray& str, stdNullPars)
            {
                require(blockExceptionsVoid(configSupport->editConfig({str.ptr, str.size}, configLoader)));
                returnTrue;
            };

            auto configEditor = cfgVarsImpl::stringReceiverByLambda(configEditorLambda);

            ////

            configFile.saveVars(serialization, false);

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

    UserPoint userPoint{false, point(0), false, false};

    {
        debugBridge::UserPoint up{};
        require(blockExceptionsVoid(up = videoOverlay->getUserPoint()));

        userPoint.valid = up.valid;
        userPoint.position = point(up.pos.X, up.pos.Y);
    }

    ////

    userPoint.signal = overview.mouseSignal;
    userPoint.signalAlt = overview.mouseSignalAlt;

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
