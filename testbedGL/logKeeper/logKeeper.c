#include "logKeeper.h"

#include "cfgTools/boolSwitch.h"
#include "cfgTools/numericVar.h"
#include "compileTools/blockExceptionsSilent.h"
#include "errorLog/errorLog.h"
#include "formattedOutput/formatters/messageFormatterImpl.h"
#include "formattedOutput/userOutputThunks.h"
#include "interfaces/fileTools.h"
#include "lib/logToBuffer/logToBuffer.h"
#include "numbers/float/floatType.h"
#include "processTools/runAndWaitProcess.h"
#include "setThreadName/setThreadName.h"
#include "stl/stlArray.h"
#include "storage/rememberCleanup.h"
#include "timer/timerKit.h"
#include "timerImpl/timerImpl.h"
#include "userOutput/printMsg.h"
#include "userOutput/printMsgEx.h"
#include "simpleString/simpleString.h"
#include "binaryFile/binaryFileImpl.h"

namespace logKeeper {

//================================================================
//
// LogKeeperImpl
//
//================================================================

struct LogKeeperImpl : public LogKeeper
{

    //----------------------------------------------------------------
    //
    // Log.
    //
    //----------------------------------------------------------------

    virtual void serialize(const CfgSerializeKit& kit);

    //----------------------------------------------------------------
    //
    // Init.
    //
    //----------------------------------------------------------------

    virtual stdbool init(const InitArgs& args, stdPars(InitKit));

    //----------------------------------------------------------------
    //
    // Run.
    //
    //----------------------------------------------------------------

    virtual void run(const RunArgs& args);

    //----------------------------------------------------------------
    //
    // LogKeeper cycle.
    //
    //----------------------------------------------------------------

    struct ShutdownReq {bool request = false;};

    void processingCycle(const RunArgs& args, ShutdownReq& shutdown);

    ////

    using CycleDiagKit = KitCombine<DiagnosticKit, TimerKit>;

    stdbool processDiag(const RunArgs& args, ShutdownReq& shutdown, stdPars(CycleDiagKit));

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
    UniqueInstance<LogBuffer> specialLogBuffer;

    ////

    logService::BufferInstances inputUpdateInstances;
    logService::BufferRefs inputUpdates = inputUpdateInstances.refs;

    ////

    SimpleString logFilename;
    UniqueInstance<TextBuffer> logMemory;

    bool fileUpdatingDisabled = false;
    OptionalObject<TimeMoment> firstUnsavedUpdate;
    OptionalObject<TimeMoment> lastAutoSave;

    BoolVar clearOnStart{true};
    NumericVar<float32> commitDelay{0, typeMax<float32>(), 1};

    BoolVar waitingReport{false};
    BoolVar savingReport{false};
    BoolVar editingReport{false};

};

////

UniquePtr<LogKeeper> LogKeeper::create()
    {return makeUnique<LogKeeperImpl>();}

//================================================================
//
// LogKeeperImpl::serialize
//
//================================================================

void LogKeeperImpl::serialize(const CfgSerializeKit& kit)
{
    CFG_NAMESPACE("Log Keeper");

    clearOnStart.serialize(kit, STR("Clear Log On Start"));
    commitDelay.serialize(kit, STR("Commit Delay In Seconds"));

    {
        CFG_NAMESPACE("Diagnostics");

        waitingReport.serialize(kit, STR("Waiting Report"));
        savingReport.serialize(kit, STR("Saving Report"));
        editingReport.serialize(kit, STR("Editing Report"));
    }
}

//================================================================
//
// LogKeeperImpl::init
//
//================================================================

stdbool LogKeeperImpl::init(const InitArgs& args, stdPars(InitKit))
{
    require(formatterArray.realloc(8192, stdPass));

    //----------------------------------------------------------------
    //
    // Expand file name.
    //
    //----------------------------------------------------------------

    logFilename.clear();

    auto getFullname = fileTools::GetString::O | [&] (auto& str)
        {logFilename = str; return def(logFilename);};

    REQUIRE(fileTools::expandPath(args.baseFilename, getFullname));

    //----------------------------------------------------------------
    //
    // Optionally clear the file.
    //
    //----------------------------------------------------------------

    BinaryFileImpl file;

    require(file.open(logFilename.str(), true, true, stdPass));

    if (clearOnStart)
        require(file.truncate(stdPass));

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    initialized = true;

    returnTrue;
}

//================================================================
//
// LogKeeperImpl::run
//
//================================================================

void LogKeeperImpl::run(const RunArgs& args)
{
    setThreadName(STR("~LogKeeper"));

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
// LogKeeperImpl::processingCycle
//
//================================================================

void LogKeeperImpl::processingCycle(const RunArgs& args, ShutdownReq& shutdown)
{

    //----------------------------------------------------------------
    //
    // Initialized? Should be checked before run, so no tolerance here.
    //
    //----------------------------------------------------------------

    TimerImpl timer;

    if_not (initialized)
    {
        specialLogBuffer->addMessage(STR("LogKeeper was not initialized, exiting."), msgErr, timer.moment());
        shutdownRequest->addRequest();
        args.guiService.addSpecialLogUpdate(*specialLogBuffer);
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

    auto sLogUpdater = LogUpdater::O | [&] ()
    {
        args.guiService.addSpecialLogUpdate(*specialLogBuffer);
    };

    REMEMBER_CLEANUP(sLogUpdater()); // Update on exit in any case.

    ////

    LogToBufferThunk msgLog{{*specialLogBuffer, formatter, timer, msgWarn, sLogUpdater}};

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
// LogKeeperImpl::processDiag
//
//================================================================

stdbool LogKeeperImpl::processDiag(const RunArgs& args, ShutdownReq& shutdown, stdPars(CycleDiagKit))
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
            returnTrue;

        ////

        auto elapsedTime = kit.timer.diff(*firstUnsavedUpdate, kit.timer.moment());
        elapsedTime = clampMin(elapsedTime, 0.f);

        ////

        auto waitTime = clampMin(commitDelay - elapsedTime, 0.f);

        ////

        if (waitTime == 0)
        {
            wait = false;
            returnTrue;
        }

        ////

        uint32 ms = 0;
        REQUIRE(convertNearest(waitTime * 1e3f, ms));
        waitTimeMs = ms;

        ////

        returnTrue;
    };

    errorBlock(setTimeout()); // Cannot exit by error before shutdown servicing.

    ////

    if (waitingReport)
    {
        if_not (waitTimeMs)
            printMsg(kit.msgLog, STR("Log keeper: Waiting infinitely."));
        else
            printMsg(kit.msgLog, STR("Log keeper: Waiting up to %s."), fltf(*waitTimeMs * 1e-3f, 3));
    }

    kit.msgLog.update();

    ////

    args.logService.takeAllUpdates(wait, waitTimeMs, inputUpdates);

    //----------------------------------------------------------------
    //
    // Save log function.
    //
    //----------------------------------------------------------------

    auto updateLogFile = [this] (stdPars(auto))
    {
        // The update is considered processed regardless of the outcome.
        REMEMBER_CLEANUP(logMemory->clear());

        ////

        auto handleFileError = [&] ()
        {
            printMsg(kit.msgLog, STR("Log keeper: File updating is stopped. To retry, press `Edit Log`."), msgWarn);
            fileUpdatingDisabled = true;
        };

        REMEMBER_CLEANUP_EX(fileErrorHandler, handleFileError());

        ////

        if_not (fileUpdatingDisabled)
        {
            BinaryFileImpl file;
            require(file.open(logFilename.str(), true, true, stdPass));
            require(file.setPosition(file.getSize(), stdPass));

            auto data = logMemory->getDataRef();

            require(file.write(data.ptr, data.size * sizeof(*data.ptr), stdPass));
        }

        ////

        fileErrorHandler.cancel();

        returnTrue;
    };

    //----------------------------------------------------------------
    //
    // Absorb a log update.
    //
    //----------------------------------------------------------------

    auto logUpdate = inputUpdates.textUpdate.hasUpdates();

    ////

    if (logUpdate)
        CHECK(logMemory->absorb(inputUpdates.textUpdate));

    //----------------------------------------------------------------
    //
    // Exit on shutdown request.
    //
    //----------------------------------------------------------------

    if (inputUpdates.shutdownRequest.hasUpdates())
    {
        shutdown.request = true;

        if (logMemory->hasUpdates())
            errorBlock(updateLogFile(stdPassNc));

        returnTrue;
    }

    //----------------------------------------------------------------
    //
    // Update the log file.
    //
    //----------------------------------------------------------------

    {
        auto currentMoment = kit.timer.moment();

        bool timeToSave =
            firstUnsavedUpdate && kit.timer.diff(*firstUnsavedUpdate, currentMoment) >= commitDelay;

        if (logUpdate && !firstUnsavedUpdate)
            firstUnsavedUpdate = currentMoment;

        if (timeToSave)
        {
            errorBlock(updateLogFile(stdPassNc));

            ////

            if (savingReport)
            {
                float32 savingTime = kit.timer.diff(currentMoment, kit.timer.moment());
                float32 elapsedTime = !lastAutoSave ? 0.f : kit.timer.diff(*lastAutoSave, currentMoment);

                printMsg(kit.msgLog, STR("Log keeper: Saving % took %s, interval %s."),
                    logFilename, fltf(savingTime, 3), fltf(elapsedTime, 3));
            }

            ////

            lastAutoSave = currentMoment;
            firstUnsavedUpdate = {};
        }
    }

    //----------------------------------------------------------------
    //
    // Edit log.
    //
    //----------------------------------------------------------------

    auto editLog = [&] ()
    {
        auto& editRequest = inputUpdates.editRequest;

        if_not (editRequest.hasUpdates())
            returnTrue;

        REMEMBER_CLEANUP(editRequest.reset());

        //
        // Save.
        //

        if (firstUnsavedUpdate || fileUpdatingDisabled)
        {
            fileUpdatingDisabled = false;

            require(updateLogFile(stdPass));
        }

        //
        // Launch editor.
        //

        if (editingReport)
        {
            printMsg(kit.msgLog, STR("Log keeper: Editing %..."), logFilename);
            kit.msgLog.update();
        }

        auto launchEditor = [&] (const auto& editor, stdPars(auto))
        {
            SimpleString cmdLine;
            cmdLine << editor << CT(" \"") << logFilename << CT("\"");
            REQUIRE(def(cmdLine));

            require(runAndWaitProcess(cmdLine.cstr(), stdPass));
            returnTrue;
        };

        if_not (errorBlock(launchEditor(editRequest.getEditor(), stdPassNc)))
        {
            printMsg(kit.msgLog, editRequest.getHelpMessage(), msgWarn);
            returnFalse;
        }

        if (editingReport)
            printMsg(kit.msgLog, STR("Log keeper: Editing finished."), logFilename);

        ////

        returnTrue;
    };

    errorBlock(editLog());

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
