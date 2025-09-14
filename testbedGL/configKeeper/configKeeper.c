#include "configKeeper.h"

#include "compileTools/blockExceptionsSilent.h"
#include "cfgVars/cfgOperations/cfgOperations.h"
#include "errorLog/errorLog.h"
#include "formattedOutput/formatters/messageFormatterImpl.h"
#include "formattedOutput/userOutputThunks.h"
#include "interfaces/fileTools.h"
#include "lib/logToBuffer/logToBuffer.h"
#include "setThreadName/setThreadName.h"
#include "stl/stlArray.h"
#include "storage/rememberCleanup.h"
#include "timer/timerKit.h"
#include "timerImpl/timerImpl.h"
#include "userOutput/printMsg.h"
#include "userOutput/printMsgEx.h"
#include "numbers/float/floatType.h"
#include "cfgTools/numericVar.h"
#include "cfgTools/boolSwitch.h"
#include "processTools/runAndWaitProcess.h"

namespace configKeeper {

//================================================================
//
// ConfigKeeperImpl
//
//================================================================

struct ConfigKeeperImpl : public ConfigKeeper
{

    //----------------------------------------------------------------
    //
    // Config.
    //
    //----------------------------------------------------------------

    virtual void serialize(const CfgSerializeKit& kit);

    //----------------------------------------------------------------
    //
    // Init.
    //
    //----------------------------------------------------------------

    virtual void init(const InitArgs& args, stdPars(InitKit));

    //----------------------------------------------------------------
    //
    // Run.
    //
    //----------------------------------------------------------------

    virtual void run(const RunArgs& args);

    //----------------------------------------------------------------
    //
    // ConfigKeeper cycle.
    //
    //----------------------------------------------------------------

    struct ShutdownReq {bool request = false;};

    void processingCycle(const RunArgs& args, ShutdownReq& shutdown);

    ////

    using CycleDiagKit = KitCombine<DiagnosticKit, TimerKit>;

    void processDiag(const RunArgs& args, ShutdownReq& shutdown, stdPars(CycleDiagKit));

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

    bool initialized = false;

    ////

    StlArray<CharType> formatterArray;

    ////

    UniqueInstance<ShutdownBuffer> shutdownRequest;
    UniqueInstance<LogBuffer> globalLogBuffer;

    ////

    configService::BufferInstances inputUpdateInstances;
    configService::BufferRefs inputUpdates = inputUpdateInstances.refs;

    ////

    SimpleString configFilename;
    UniqueInstance<CfgTree> configMemory;
    UniqueInstance<CfgOperations> cfgOperations;

    bool fileUpdatingDisabled = false;
    OptionalObject<TimeMoment> firstUnsavedUpdate;
    OptionalObject<TimeMoment> lastAutoSave;
    NumericVar<float32> commitDelay{0, typeMax<float32>(), 1};
    BoolVar waitingReport{false};
    BoolVar savingReport{false};
    BoolVar editingReport{false};

    ////

    BoolVar memReport{false};
    size_t memReported = 0;

};

////

UniquePtr<ConfigKeeper> ConfigKeeper::create()
    {return makeUnique<ConfigKeeperImpl>();}

//================================================================
//
// ConfigKeeperImpl::serialize
//
//================================================================

void ConfigKeeperImpl::serialize(const CfgSerializeKit& kit)
{
    CFG_NAMESPACE("Config Keeper");

    commitDelay.serialize(kit, STR("Commit Delay In Seconds"));

    {
        CFG_NAMESPACE("Diagnostics");

        memReport.serialize(kit, STR("Memory Report"));
        waitingReport.serialize(kit, STR("Waiting Report"));
        savingReport.serialize(kit, STR("Saving Report"));
        editingReport.serialize(kit, STR("Editing Report"));
    }
}

//================================================================
//
// ConfigKeeperImpl::init
//
//================================================================

void ConfigKeeperImpl::init(const InitArgs& args, stdPars(InitKit))
{
    formatterArray.realloc(4096, stdPass);

    //----------------------------------------------------------------
    //
    // Load and update config file for ALL subsystems
    // while they are in stopped state.
    //
    //----------------------------------------------------------------

    {
        configFilename.clear();

        auto getFullname = fileTools::GetString::O | [&] (auto& str)
            {configFilename = str; return def(configFilename);};

        REQUIRE(fileTools::expandPath(args.baseFilename, getFullname));
    }

    //----------------------------------------------------------------
    //
    // Load config.
    //
    //----------------------------------------------------------------

    bool configFound = fileTools::isFile(configFilename.cstr());

    if (configFound)
    {
        cfgOperations->loadFromFile(*configMemory, configFilename.cstr(), false, stdPass);
        cfgOperations->loadVars(*configMemory, args.serialization, {false, true}, stdPass);
    }

    //
    // Save all vars, update the file.
    //

    cfgOperations->saveVars(*configMemory, args.serialization, {false, true}, stdPass);

    cfgOperations->saveToFile(*configMemory, configFilename.cstr(), stdPass);

    if_not (configFound)
        printMsg(kit.msgLog, STR("Created config file %"), configFilename, msgWarn);

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    initialized = true;
}

//================================================================
//
// ConfigKeeperImpl::run
//
//================================================================

void ConfigKeeperImpl::run(const RunArgs& args)
{
    setThreadName(STR("~ConfigKeeper"));

    ////

    for (; ;)
    {
        blockExceptBegin;

        ////

        ShutdownReq shutdown;

        processingCycle(args, shutdown);

        if (shutdown.request)
            break;

        ////

        blockExceptEndIgnore;
    }
}

//================================================================
//
// ConfigKeeperImpl::processingCycle
//
//================================================================

void ConfigKeeperImpl::processingCycle(const RunArgs& args, ShutdownReq& shutdown)
{

    //----------------------------------------------------------------
    //
    // Initialized? Should be checked before run, so no tolerance here.
    //
    //----------------------------------------------------------------

    TimerImpl timer;

    if_not (initialized)
    {
        globalLogBuffer->addMessage(STR("ConfigKeeper was not initialized, exiting."), msgErr, timer.moment());
        shutdownRequest->addRequest();
        args.guiService.addGlobalLogUpdate(*globalLogBuffer);
        args.guiService.addShutdownRequest(*shutdownRequest);
        shutdown.request = true;
        return;
    }

    //----------------------------------------------------------------
    //
    // Message output.
    //
    //----------------------------------------------------------------

    MessageFormatterImpl formatter{formatterArray};

    ////

    auto gLogUpdater = LogUpdater::O | [&] ()
    {
        args.guiService.addGlobalLogUpdate(*globalLogBuffer);
    };

    REMEMBER_CLEANUP(gLogUpdater()); // Update on exit in any case.

    ////

    LogToBufferThunk msgLog{{*globalLogBuffer, formatter, timer, msgWarn, gLogUpdater}};

    ErrorLogByMsgLog errorLog(msgLog);
    ErrorLogKit errorLogKit(errorLog);

    MsgLogExByMsgLog msgLogEx(msgLog);


    MsgLogNull msgLogNull;

    auto kit = kitCombine
    (
        TimerKit{timer},
        MessageFormatterKit{formatter},
        MsgLogKit{msgLog},
        ErrorLogKit{errorLog},
        MsgLogExKit{msgLogEx}
    );

    ////

    stdTraceRoot;

    errorBlock(processDiag(args, shutdown, stdPassNc));

}

//================================================================
//
// ConfigKeeperImpl::processDiag
//
//================================================================

void ConfigKeeperImpl::processDiag(const RunArgs& args, ShutdownReq& shutdown, stdPars(CycleDiagKit))
{

    //----------------------------------------------------------------
    //
    // Wait for new updates.
    //
    //----------------------------------------------------------------

    bool wait = true;
    OptionalObject<uint32> waitTimeMs;

    ////

    auto setTimeout = [&] ()
    {
        if_not (firstUnsavedUpdate)
            return;

        ////

        auto elapsedTime = kit.timer.diff(*firstUnsavedUpdate, kit.timer.moment());
        elapsedTime = clampMin(elapsedTime, 0.f);

        ////

        auto waitTime = clampMin(commitDelay - elapsedTime, 0.f);

        ////

        if (waitTime == 0)
        {
            wait = false;
            return;
        }

        ////

        uint32 ms = 0;
        REQUIRE(convertNearest(waitTime * 1e3f, ms));
        waitTimeMs = ms;
    };

    errorBlock(setTimeout()); // Cannot exit by error before shutdown servicing.

    ////

    if (waitingReport)
    {
        if_not (waitTimeMs)
            printMsg(kit.msgLog, STR("Config keeper: Waiting infinitely."));
        else
            printMsg(kit.msgLog, STR("Config keeper: Waiting up to %s."), fltf(*waitTimeMs * 1e-3f, 3));
    }

    kit.msgLog.update();

    ////

    configService::BoardDiagnostics diagnostics;
    args.configService.takeAllUpdates(wait, waitTimeMs, inputUpdates, diagnostics);

    //----------------------------------------------------------------
    //
    // Save config function.
    //
    //----------------------------------------------------------------

    auto saveConfig = [this] (stdPars(auto))
    {
        auto handleFileError = [&] ()
        {
            printMsg(kit.msgLog, STR("Config keeper: File updating is stopped. To retry, press `Edit Config`."), msgWarn);
            fileUpdatingDisabled = true;
        };

        REMEMBER_CLEANUP_EX(fileErrorHandler, handleFileError());

        if_not (fileUpdatingDisabled)
        {
            cfgOperations->saveToFile(*configMemory, configFilename.cstr(), stdPass);
        }

        fileErrorHandler.cancel();
    };

    //----------------------------------------------------------------
    //
    // Exit on shutdown request.
    //
    //----------------------------------------------------------------

    if (inputUpdates.shutdownRequest.hasUpdates())
    {
        shutdown.request = true;

        if (firstUnsavedUpdate)
            errorBlock(saveConfig(stdPassNc));

        return;
    }

    //----------------------------------------------------------------
    //
    // Print memory usage.
    //
    //----------------------------------------------------------------

    if (memReport)
    {
        auto memory =
            configMemory->allocatedBytes() +
            inputUpdates.configUpdate.allocatedBytes() +
            diagnostics.configBufferBytes;

        if_not (memory <= memReported)
        {
            printMsg
            (
                kit.msgLog, STR("Config keeper: Buffers % KB (% KB)"),
                fltf(1e-3f * memory, 3),
                fltfs(1e-3f * memory - 1e-3f * memReported, 3)
            );
        }

        memReported = memory;
    }

    //----------------------------------------------------------------
    //
    // Absorb a config update into the main config tree.
    //
    //----------------------------------------------------------------

    auto configUpdate = inputUpdates.configUpdate.hasUpdates();

    ////

    if (configUpdate)
        CHECK(configMemory->absorb(inputUpdates.configUpdate));

    //----------------------------------------------------------------
    //
    // Update the config file.
    //
    //----------------------------------------------------------------

    {
        auto currentMoment = kit.timer.moment();

        bool timeToSave =
            firstUnsavedUpdate && kit.timer.diff(*firstUnsavedUpdate, currentMoment) >= commitDelay;

        if (configUpdate && !firstUnsavedUpdate)
            firstUnsavedUpdate = currentMoment;

        if (timeToSave)
        {
            errorBlock(saveConfig(stdPassNc));

            ////

            if (savingReport)
            {
                float32 savingTime = kit.timer.diff(currentMoment, kit.timer.moment());
                float32 elapsedTime = !lastAutoSave ? 0.f : kit.timer.diff(*lastAutoSave, currentMoment);

                printMsg(kit.msgLog, STR("Config keeper: Saving % took %s, interval %s."),
                    configFilename, fltf(savingTime, 3), fltf(elapsedTime, 3));
            }

            ////

            lastAutoSave = currentMoment;
            firstUnsavedUpdate = {};
        }
    }

    //----------------------------------------------------------------
    //
    // Edit config.
    //
    //----------------------------------------------------------------

    auto editConfig = [&] ()
    {
        auto& editRequest = inputUpdates.editRequest;

        if_not (editRequest.hasUpdates())
            return;

        REMEMBER_CLEANUP(editRequest.reset());

        //
        // Save.
        //

        if (firstUnsavedUpdate || fileUpdatingDisabled)
        {
            fileUpdatingDisabled = false;

            saveConfig(stdPass);
        }

        //
        // Launch editor.
        //

        if (editingReport)
        {
            printMsg(kit.msgLog, STR("Config keeper: Editing %..."), configFilename);
            kit.msgLog.update();
        }

        auto launchEditor = [&] (const auto& editor, stdPars(auto))
        {
            SimpleString cmdLine;
            cmdLine << editor << CT(" \"") << configFilename << CT("\"");
            REQUIRE(def(cmdLine));

            runAndWaitProcess(cmdLine.cstr(), stdPass);
        };

        if_not (errorBlock(launchEditor(editRequest.getEditor(), stdPassNc)))
        {
            printMsg(kit.msgLog, editRequest.getHelpMessage(), msgWarn);
            returnFalse;
        }

        if (editingReport)
            printMsg(kit.msgLog, STR("Config keeper: Editing finished."), configFilename);

        //
        // Clear all flags.
        // Load vars from file,
        // Generate an update for changed vars.
        //

        configMemory->clearAllDataChangedFlags();

        cfgOperations->loadFromFile(*configMemory, configFilename.cstr(), true, stdPass);

        UniqueInstance<CfgTree> userUpdate; // Editing is rare, always allocate.
        configMemory->generateUpdate(*userUpdate);

        //
        // Serialize the update to itsef.
        //

        auto serialization = cfgSerializationThunk | [&] (auto& kit) {serialize(kit);};
        cfgOperations->loadVars(*userUpdate, serialization, {false, true}, stdPass);

        //
        // Send the update to the GUI.
        //

        REQUIRE(args.guiService.addConfigUpdate(*userUpdate));
    };

    errorBlock(editConfig());
}

//----------------------------------------------------------------

}
