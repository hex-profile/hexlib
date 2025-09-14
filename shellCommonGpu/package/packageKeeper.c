#include "packageKeeper.h"

#include <stdexcept>

#include "minimalShell/minimalShell.h"
#include "cfgVars/configFile/configFile.h"
#include "debugBridge/bridgeUsage/actionReceivingByBridge.h"
#include "debugBridge/bridgeUsage/actionSetupToBridge.h"
#include "debugBridge/bridgeUsage/baseImageConsoleToBridge.h"
#include "storage/rememberCleanup.h"
#include "dataAlloc/arrayMemory.h"
#include "signalsTools/legacySignalsImpl.h"
#include "userOutput/printMsgEx.h"
#include "overlayTakeover/overlayTakeoverThunk.h"
#include "storage/adapters/memberLambda.h"
#include "imageRead/positionTools.h"
#include "kits/userPoint.h"
#include "interfaces/fileTools.h"

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

    void init(const CharType* const configName, SerializeTarget& target, stdPars(StarterDebugKit));
    void finalize(SerializeTarget& target, stdPars(StarterDebugKit));

    //----------------------------------------------------------------
    //
    // Process helper.
    //
    //----------------------------------------------------------------

public:

    void process(ProcessTarget& target, bool warmup, stdPars(StarterDebugKit));

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

    UniqueInstance<MinimalShell> shell;

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
    UniqueInstance<ConfigFile> configFilePtr;
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
    {
        CFG_NAMESPACE("~Shell");

        shell->serialize(kit);

        deactivateOverlay.serialize(kit, STR("Deactivate Overlay"), STR("\\"), STR("Deactivate Overlay"));

        {
            CFG_NAMESPACE("Engine Memory");
            engineMemory.serialize(kit);
        }
    }
}

//================================================================
//
// PackageKeeperImpl::init
//
//================================================================

void PackageKeeperImpl::init(const CharType* const configName, SerializeTarget& target, stdPars(StarterDebugKit))
{

    REQUIRE(!initialized);

    //----------------------------------------------------------------
    //
    // Config file.
    //
    //----------------------------------------------------------------

    auto targetLambda = memberLambda(target, &SerializeTarget::serialize);
    auto serialization = overlaySerializationThunk(targetLambda, overlayOwnerID);

    ////

    configFileActive = configName && *configName;

    ////

    if (configFileActive)
    {
        auto action = [&] ()
        {
            bool configFound = fileTools::isFile(configName);

            ////

            configFile.loadFile(SimpleString{configName}, stdPass);
            configFile.loadVars(serialization, true, stdPass);

            ////

            configFile.saveVars(serialization, true, stdPass);
            configFile.updateFile(true, stdPass);

            ////

            if_not (configFound)
                printMsg(kit.msgLog, STR("Created config file %"), configName, msgWarn);
        };

        errorBlock(action());
    }

    //----------------------------------------------------------------
    //
    // Register signals.
    //
    //----------------------------------------------------------------

    auto bridgeActionSetup = kit.debugBridge.actionSetup();
    REQUIRE(bridgeActionSetup);

    BaseActionSetupToBridge actionSetup{*bridgeActionSetup, stdPassNc};

    ////

    REMEMBER_CLEANUP_EX(signalsCleanup, {actionSetup.actsetClear(); signalHist.dealloc(); signalMousePos = BaseMousePos{};});

    ////

    require(actionSetup.actsetClear());

    ////

    int32 signalCount{};
    signalImpl::registerSignals(serialization, actionSetup, signalCount, stdPass);

    ////

    require(actionSetup.actsetUpdate());

    ////

    signalHist.realloc(signalCount, cpuBaseByteAlignment, kit.malloc, stdPass);

    //----------------------------------------------------------------
    //
    // Shell init.
    //
    //----------------------------------------------------------------

    shell->init(stdPass);

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    initialized = true;

    signalsCleanup.cancel();
}

//================================================================
//
// PackageKeeperImpl::finalize
//
//================================================================

void PackageKeeperImpl::finalize(SerializeTarget& target, stdPars(StarterDebugKit))
{
    REQUIRE(initialized);

    //----------------------------------------------------------------
    //
    // Save config if active.
    //
    //----------------------------------------------------------------

    auto targetLambda = memberLambda(target, &SerializeTarget::serialize);
    auto serialization = overlaySerializationThunk(targetLambda, overlayOwnerID);

    if (configFileActive)
    {
        errorBlock(configFile.saveVars(serialization, true, stdPassNc)) &&
        errorBlock(configFile.updateFile(false, stdPassNc));
    }

    //----------------------------------------------------------------
    //
    // Make profiler report if profiling is active.
    //
    //----------------------------------------------------------------

    if (shell->profilingActive())
        errorBlock(shell->profilingReport(nullptr, stdPassNc));
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

    virtual void realloc(stdPars(EngineReallocKit))
        {base.realloc(stdPassThru);}

public:

    virtual void process(stdPars(EngineProcessKit))
    {
        auto modifiedKits = kitCombine
        (
            ProfilerKit(!warmup ? kit.profiler : nullptr),
            VerbosityKit(!warmup ? kit.verbosity : Verbosity::Off)
        );

        base.process(stdPassThruKit(kitReplace(kit, modifiedKits)));
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

void PackageKeeperImpl::process(ProcessTarget& target, bool warmup, stdPars(StarterDebugKit))
{
    REQUIRE(initialized);

    auto targetLambda = [&] (const auto& kit) {target.serialize(kit);};
    auto serialization = overlaySerializationThunk(targetLambda, overlayOwnerID);

    //----------------------------------------------------------------
    //
    // Handle signals.
    //
    //----------------------------------------------------------------

    auto bridgeReceiving = kit.debugBridge.actionReceiving();
    REQUIRE(bridgeReceiving);
    BaseActionReceivingByBridge actionReceiving{*bridgeReceiving, stdPassNc};

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

        BaseActionSetupToBridge actionSetup{*bridgeActionSetup, stdPassNc};

        ////

        REMEMBER_CLEANUP_EX(signalsCleanup, {actionSetup.actsetClear(); signalHist.dealloc();});

        ////

        printMsgL(kit, STR("Actions resetup"), msgWarn);

        ////

        require(actionSetup.actsetClear());

        ////

        int32 signalCount{};
        signalImpl::registerSignals(serialization, actionSetup, signalCount, stdPass);

        ////

        require(actionSetup.actsetUpdate());

        ////

        signalHist.realloc(signalCount, cpuBaseByteAlignment, kit.malloc, stdPass);

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
            auto action = [&] ()
            {
                configFile.loadFromString(charArray(config.ptr, config.size), stdPass);
                configFile.loadVars(serialization, true, stdPass);
                configFile.updateFile(true, stdPass);
            };

            if_not (errorBlock(action()))
                throw std::runtime_error("Config support error"); // Debug bridge uses exceptions.
        };

        ////

        auto configLoader = db::configReceiverByLambda(configLoaderLambda);

        //
        // Update config vars. Also saves to disk if dirty.
        //

        auto configUpdateVars = [&] (stdPars(auto))
        {
            configFile.saveVars(serialization, false, stdPass);
            configFile.updateFile(false, stdPass);
        };

        //
        // Save config operation.
        //

        if (overview.saveConfig)
        {
            auto configSaver = cfgVarsImpl::StringReceiver::O | [&] (const CharArray& str, stdParsNull)
            {
                stdExceptBegin;
                configSupport->saveConfig({str.ptr, str.size});
                stdExceptEnd;
            };

            ////

            configUpdateVars(stdPass);

            configFile.saveToString(configSaver, stdPass);
        }

        //
        // Load config operation.
        //

        if (overview.loadConfig)
        {
            convertExceptions(configSupport->loadConfig(configLoader));
        }

        //
        // Edit config operation.
        //

        if (overview.editConfig)
        {
            auto configEditor = cfgVarsImpl::StringReceiver::O | [&] (const CharArray& str, stdParsNull)
            {
                stdExceptBegin;
                configSupport->editConfig({str.ptr, str.size}, configLoader);
                stdExceptEnd;
            };

            ////

            configUpdateVars(stdPass);

            configFile.saveToString(configEditor, stdPass);
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
        convertExceptions(videoOverlay->clear());

    ////

    UserPoint userPoint;

    if (overview.mousePos.valid())
        signalMousePos = overview.mousePos;

    userPoint.valid = signalMousePos.valid();
    userPoint.floatPos = convertIndexToPos(signalMousePos.pos());

    userPoint.leftSet = (overview.mouseLeftSet != 0);
    userPoint.rightSet = (overview.mouseRightSet != 0);

    userPoint.leftReset = (overview.mouseLeftReset != 0);
    userPoint.rightReset = (overview.mouseRightReset != 0);

    ////

    BaseVideoOverlayToBridge baseOverlay{*kit.debugBridge.videoOverlay(), kit};

    ////

    DesiredOutputSize desiredOutputSize;

    auto kitEx = kitCombine
    (
        kit,
        minimalShell::BaseImageConsolesKit{nullptr, nullptr, &baseOverlay},
        UserPointKit{userPoint},
        DesiredOutputSizeKit{desiredOutputSize}
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

    shell->process({nullptr, targetEx, engineMemory, true, sysAllocHappened}, stdPassKit(kitEx));

    if (sysAllocHappened && !warmup)
        printMsgL(kit, STR("WARNING: System realloc."), msgWarn);
}

//----------------------------------------------------------------

}
}
