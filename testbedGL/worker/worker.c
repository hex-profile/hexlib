#include "worker.h"

#include <thread>
#include <chrono>

#include "allocation/mallocAllocator/mallocAllocator.h"
#include "allocation/mallocKit.h"
#include "cfgTools/boolSwitch.h"
#include "cfgTools/numericVar.h"
#include "cfgVars/cfgSerializeImpl/cfgSerializeImpl.h"
#include "compileTools/blockExceptionsSilent.h"
#include "dataAlloc/arrayObjectMemory.inl"
#include "errorLog/convertExceptions.h"
#include "errorLog/debugBreak.h"
#include "formattedOutput/formatters/messageFormatterImpl.h"
#include "formattedOutput/requireMsg.h"
#include "formattedOutput/userOutputThunks.h"
#include "formatting/messageFormatterKit.h"
#include "gpuLayer/gpuLayerImpl.h"
#include "imageRead/positionTools.h"
#include "kits/userPoint.h"
#include "lib/logToBuffer/logToBuffer.h"
#include "minimalShell/minimalShell.h"
#include "numbers/int/intType.h"
#include "overlayTakeover/overlayTakeoverThunk.h"
#include "setThreadName/setThreadName.h"
#include "signalsTools/signalTools.h"
#include "stl/stlArray.h"
#include "storage/rememberCleanup.h"
#include "tests/testShell/testShell.h"
#include "threads/threads.h"
#include "timer/timerKit.h"
#include "timerImpl/timerImpl.h"
#include "userOutput/diagnosticKitNull.h"
#include "userOutput/printMsg.h"
#include "userOutput/printMsgEx.h"
#include "worker/gpuBaseConsoleThunk.h"

namespace worker {

using namespace std;
using minimalShell::MinimalShell;

//================================================================
//
// WorkerImpl
//
//================================================================

struct WorkerImpl : public Worker
{

    WorkerImpl(UniquePtr<TestModule>&& testModule)
        : testModule{move(testModule)}
    {
    }

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
    // Worker cycle.
    //
    //----------------------------------------------------------------

    struct ShutdownReq {bool request = false;};

    struct RunArgsEx : public RunArgs
    {
        bool& glContextBound;
        ShutdownReq& shutdown;

        RunArgsEx(const RunArgs& base, bool& glContextBound, ShutdownReq& shutdown)
            : RunArgs{base}, glContextBound{glContextBound}, shutdown{shutdown} {}
    };

    void processingCycle(const RunArgsEx& args);

    ////

    using CycleDiagKit = KitCombine<DiagnosticKit, LocalLogKit, LocalLogAuxKit, TimerKit, MallocKit>;

    void processDiag(const RunArgsEx& args, stdPars(CycleDiagKit));

    //----------------------------------------------------------------
    //
    // GL context: Before the test module.
    //
    //----------------------------------------------------------------

    ContextBinder* glContext = nullptr;

    //----------------------------------------------------------------
    //
    // Minimal shell, test module and its memory: the order of construction is important.
    //
    //----------------------------------------------------------------

    UniqueInstance<MinimalShell> minimalShell;

    void minimalShellInit()
    {
        minimalShell->settings().setGpuContextMaintainer(false);
    }

    int minimalShellInitCaller = (minimalShellInit(), 0);

    ////

    MemController testModuleMemory;

    ////

    UniqueInstance<TestShell> testShell;

    ////

    UniquePtr<TestModule> testModule;

    //----------------------------------------------------------------
    //
    // Config.
    //
    //----------------------------------------------------------------

    OverlayTakeoverID overlayOwnerID = OverlayTakeoverID::undefined();

    StandardSignal deactivateOverlay;

    ////

    StandardSignal processStep;

    BoolSwitch processContinuously{false};

    ////

    NumericVar<int> gLogDebuggerOutputLevel{0, typeMax<int>(), msgErr};

    NumericVar<int> lLogDebuggerOutputLevel{0, typeMax<int>(), msgErr + 1};

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

    bool initialized = false;

    //----------------------------------------------------------------
    //
    // Vars.
    //
    //----------------------------------------------------------------

    StlArray<CharType> formatterArray;
    ArrayObjectMemory<int32> actionHist;

    ////

    workerService::BufferInstances inputUpdateInstances;
    workerService::BufferRefs inputUpdates = inputUpdateInstances.refs;

    UniqueInstance<DisplaySettingsBuffer> currentDisplaySettings;
    OptionalObject<Point<float32>> currentMousePos;

    ////

    guiService::BufferInstances outputBufferInstances;
    guiService::BufferRefs outputBuffers = outputBufferInstances.refs;

    ////

    UniqueInstance<CfgTree> configOutputBuffer;
    cfgVarsImpl::CfgTemporary configTemp;

    //----------------------------------------------------------------
    //
    // Debug.
    //
    //----------------------------------------------------------------

    NumericVar<int> workerCycleDelayMs{0, typeMax<int>(), 0};
    int32 cycleCounter = 0;
    StandardSignal debugErrorSignal;

};

////

UniquePtr<Worker> Worker::create(UniquePtr<TestModule>&& testModule)
    {return makeUnique<WorkerImpl>(move(testModule));}

//================================================================
//
// WorkerImpl::serialize
//
//================================================================

void WorkerImpl::serialize(const CfgSerializeKit& kit)
{
    {
        CFG_NAMESPACE("Worker");

        deactivateOverlay.serialize(kit, STR("Deactivate Overlay"), STR("\\"), STR("Deactivate Overlay"));

        processContinuously.serialize(kit, STR("Process Continuously"), STR("F5"), STR("Process Continuously"));

        if_not (processStep.serialize(kit, STR("Process Step"), STR("F8"), STR("Process Step")))
            processContinuously = false;

        {
            CFG_NAMESPACE("Global Log");
            gLogDebuggerOutputLevel.serialize(kit, STR("Debugger Output Level"), STR("0 info, 1 warnings, 2 errors, 3 nothing"));
        }

        {
            CFG_NAMESPACE("Local Log");
            lLogDebuggerOutputLevel.serialize(kit, STR("Debugger Output Level"), STR("0 info, 1 warnings, 2 errors, 3 nothing"));
        }

        {
            CFG_NAMESPACE("Debug");

            debugErrorSignal.serialize(kit, STR("Generate Test Error"));
            workerCycleDelayMs.serialize(kit, STR("Cycle Delay In Milliseconds"));
        }
    }

    {
        CFG_NAMESPACE("Algorithm Shell");
        minimalShell->serialize(kit);

        {
            CFG_NAMESPACE("Algorithm Module Memory");
            testModuleMemory.serialize(kit);
        }
    }

    {
        OverlayTakeoverThunk overlayTakeover{overlayOwnerID};

        auto& oldKit = kit;
        auto kit = kitCombine(oldKit, OverlayTakeoverKit{overlayTakeover});

        testShell->serialize(kit);

        testModule->serialize(kit);
    }
}

//================================================================
//
// WorkerImpl::init
//
//================================================================

void WorkerImpl::init(const InitArgs& args, stdPars(InitKit))
{
    formatterArray.realloc(65536, stdPass);

    //----------------------------------------------------------------
    //
    // OpenGL context
    //
    //----------------------------------------------------------------

    glContext = &args.glContext;

    glContext->bind(stdPass);
    REMEMBER_CLEANUP_ERROR_BLOCK(glContext->unbind(stdPassNc));

    //----------------------------------------------------------------
    //
    // Minimal shell.
    //
    //----------------------------------------------------------------

    minimalShell->init(stdPass);

    //----------------------------------------------------------------
    //
    // Create module externals.
    //
    //----------------------------------------------------------------

    UniquePtr<TestModuleExternals> externals;

    convertExceptions(externals = args.externalsFactory());

    testModule->setExternals(move(externals));

    //----------------------------------------------------------------
    //
    // Update action set.
    //
    //----------------------------------------------------------------

    REMEMBER_CLEANUP_EX(updateActionsErrMsg, printMsg(kit.msgLog, STR("WORKER: Failed to update action set."), msgErr));

    outputBuffers.actionSetUpdate.dataClear();

    ////

    size_t signalCount{};

    {
        bool actionsAddOk = true;

        auto receiver = signalTools::ActionReceiver::O | [&] (ActionId id, CharArray name, CharArray key, CharArray comment)
        {
            check_flag(outputBuffers.actionSetUpdate.dataAdd(id, name, key, comment), actionsAddOk);
        };

        auto serialization = cfgSerializationThunk | [&] (auto& kit) {serialize(kit);};

        signalTools::gatherActionSet(serialization, receiver, signalCount, stdPass);

        REQUIRE(actionsAddOk);
    }

    ////

    actionHist.reallocInHeap(Space(signalCount), stdPass);

    ////

    REQUIRE(args.guiService.addActionSetUpdate(outputBuffers.actionSetUpdate));

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    updateActionsErrMsg.cancel();
    initialized = true;
}

//================================================================
//
// WorkerImpl::run
//
//================================================================

void WorkerImpl::run(const RunArgs& args)
{
    setThreadName(STR("~WORKER"));

    //----------------------------------------------------------------
    //
    // Remember to unbind GL context.
    //
    //----------------------------------------------------------------

    bool glContextBound = false;

    ////

    auto unbindContext = [&] ()
    {
        stdTraceRoot;
        DiagnosticKitNull kit;

        if (glContextBound)
        {
            if (glContext)
                errorBlock(glContext->unbind(stdPassNc));
        }
    };

    REMEMBER_CLEANUP(unbindContext());

    //----------------------------------------------------------------
    //
    // Worker loop.
    //
    //----------------------------------------------------------------

    for (; ;)
    {
        blockExceptBegin;

        ////

        ShutdownReq shutdown;
        RunArgsEx argsEx{args, glContextBound, shutdown};

        processingCycle(argsEx);

        if (shutdown.request)
            break;

        ////

        blockExceptEndIgnore;
    }
}

//================================================================
//
// WorkerImpl::processingCycle
//
//================================================================

void WorkerImpl::processingCycle(const RunArgsEx& args)
{

    //----------------------------------------------------------------
    //
    // Initialized? Should be checked before run, so no tolerance here.
    //
    //----------------------------------------------------------------

    TimerImpl timer;

    if_not (initialized)
    {
        outputBuffers.globalLogUpdate.addMessage(STR("Worker was not initialized, exiting."), msgErr, timer.moment());
        outputBuffers.shutdownRequest.addRequest();
        args.guiService.addAllUpdates(outputBuffers);
        args.shutdown.request = true;
        return;
    }

    //----------------------------------------------------------------
    //
    // Get new updates and exit on shutdown request.
    //
    //----------------------------------------------------------------

    bool wait = !processContinuously;

    args.workerService.takeAllUpdates(wait, inputUpdates);

    ////

    if (inputUpdates.shutdownRequest.hasUpdates())
    {
        args.shutdown.request = true;
        return;
    }

    //----------------------------------------------------------------
    //
    // Update GUI buffers on exit in any case.
    //
    //----------------------------------------------------------------

    REMEMBER_CLEANUP(args.guiService.addAllUpdates(outputBuffers));

    //----------------------------------------------------------------
    //
    // Actions to be taken on error:
    // * Stop continuous mode.
    //
    //----------------------------------------------------------------

    auto errorExitFunc = [&] ()
    {
        if (processContinuously)
        {
            processContinuously = false;
            outputBuffers.localLogUpdate.addMessage(STR("RUN mode stopped."), msgWarn, timer.moment());
        }
    };

    REMEMBER_CLEANUP_EX(errorExit, errorExitFunc());

    //----------------------------------------------------------------
    //
    // Clear local log and overlay.
    //
    //----------------------------------------------------------------

    outputBuffers.localLogUpdate.clearLog();
    outputBuffers.overlayUpdate.clearImage();

    //----------------------------------------------------------------
    //
    // Message output.
    //
    //----------------------------------------------------------------

    //
    // Formatter.
    //

    MessageFormatterImpl formatter{formatterArray};

    //
    // Global log.
    //

    auto gLogUpdater = LogUpdater::O | [&] ()
    {
        args.guiService.addGlobalLogUpdate(outputBuffers.globalLogUpdate);
    };

    LogToBufferThunk msgLog{{outputBuffers.globalLogUpdate, formatter, timer, gLogDebuggerOutputLevel, gLogUpdater}};

    ErrorLogByMsgLog errorLog(msgLog);
    ErrorLogKit errorLogKit(errorLog);

    MsgLogExByMsgLog msgLogEx(msgLog);

    //
    // Local log.
    //

    auto lLogUpdater = LogUpdater::O | [&] ()
    {
        args.guiService.addLocalLogUpdate(outputBuffers.localLogUpdate);
    };

    LogToBufferThunk localLog{{outputBuffers.localLogUpdate, formatter, timer, lLogDebuggerOutputLevel, lLogUpdater}};

    //----------------------------------------------------------------
    //
    // Make basic diag kit and call further.
    //
    //----------------------------------------------------------------

    MAKE_MALLOC_ALLOCATOR(errorLogKit);

    MsgLogNull msgLogNull;

    auto kit = kitCombine
    (
        TimerKit{timer},
        MallocKit{mallocAllocator},
        MessageFormatterKit{formatter},
        MsgLogKit{msgLog},
        ErrorLogKit{errorLog},
        MsgLogExKit{msgLogEx},
        LocalLogKit{localLog},
        LocalLogAuxKit{false, msgLogNull}
    );

    ////

    stdTraceRoot;

    if_not (errorBlock(processDiag(args, stdPassNc)))
        return;

    //----------------------------------------------------------------
    //
    // Success.
    //
    //----------------------------------------------------------------

    errorExit.cancel();

}

//================================================================
//
// WorkerImpl::processDiag
//
//================================================================

void WorkerImpl::processDiag(const RunArgsEx& args, stdPars(CycleDiagKit))
{
    stdScopedBegin;

    auto serialization = cfgSerializationThunk | [&] (auto& kit) {serialize(kit);};

    //----------------------------------------------------------------
    //
    // Bind GL context if it's not bound.
    //
    //----------------------------------------------------------------

    if_not (args.glContextBound)
    {
        glContext->bind(stdPass);
        args.glContextBound = true;
    }

    //----------------------------------------------------------------
    //
    // Load input config vars (sent after user edit).
    //
    //----------------------------------------------------------------

    if (inputUpdates.configUpdate.hasUpdates())
    {
        using namespace cfgSerializeImpl;
        errorBlock(loadVarsFromTree({serialization, inputUpdates.configUpdate, configTemp, false, true}, stdPassNc));
        inputUpdates.configUpdate.clearMemory(); // A rare operation, deallocate.
    }

    //----------------------------------------------------------------
    //
    // Update all signals.
    //
    // If overlay owner has changed, re-feed all the signals
    // to clean outdated switches.
    //
    //----------------------------------------------------------------

    {
        auto prevOverlayID = overlayOwnerID;

        ////

        auto& actionsUpdate = inputUpdates.actionsUpdate;

        auto provider = signalTools::ActionIdProvider::O | [&] (auto& receiver)
        {
            actionsUpdate.dataGet(receiver);
        };

        signalTools::updateSignals(actionsUpdate.hasUpdates(), provider, serialization, actionHist);

        if (deactivateOverlay)
            overlayOwnerID = OverlayTakeoverID::cancelled();

        if_not (prevOverlayID == overlayOwnerID)
            signalTools::updateSignals(actionsUpdate.hasUpdates(), provider, serialization, actionHist);
    }

    //----------------------------------------------------------------
    //
    // Remember to update changed config variables in any case.
    //
    //----------------------------------------------------------------

    auto updateConfigVars = [&] ()
    {
        using namespace cfgSerializeImpl;

        saveVarsToTree({serialization, *configOutputBuffer, configTemp, true, true, false}, stdPass);

        if (configOutputBuffer->hasUpdates())
            REQUIRE(args.configService.addConfigUpdate(*configOutputBuffer));
    };

    REMEMBER_CLEANUP(errorBlock(updateConfigVars()));

    //----------------------------------------------------------------
    //
    // Header.
    //
    //----------------------------------------------------------------

    printMsgL(kit, STR("% mode"), processContinuously ? STR("RUN") : STR("STEP"));
    // printMsgL(kit, STR("Worker Cycle %"), cycleCounter++);

    ////

    REMEMBER_CLEANUP
    (
        if (workerCycleDelayMs)
            this_thread::sleep_for(chrono::milliseconds(workerCycleDelayMs));
    );

    //----------------------------------------------------------------
    //
    // Debug error signal.
    //
    //----------------------------------------------------------------

    REQUIRE(!debugErrorSignal);

    //----------------------------------------------------------------
    //
    // Extend kit with basic GPU tools.
    //
    //----------------------------------------------------------------

    GpuInitApiImpl gpuInitApi{kit};
    auto gpuInitKit = gpuInitApi.getKit();

    auto& baseKit = kit;

    auto kit = kitCombine
    (
        baseKit,
        gpuInitKit,
        GpuPropertiesKit{args.externalContext.gpuProperties},
        GpuCurrentContextKit{args.externalContext.gpuContext},
        GpuCurrentStreamKit{args.externalContext.gpuStream}
    );

    kit.gpuContextSetting.threadContextSet(kit.gpuCurrentContext, stdPass);

    //----------------------------------------------------------------
    //
    // GPU base console.
    //
    //----------------------------------------------------------------

    auto overlayUpdater = gpuBaseConsoleThunk::OverlayUpdater::O | [&] (stdParsNull)
    {
        REQUIRE(args.guiService.addOverlayUpdate(outputBuffers.overlayUpdate));
    };

    GpuBaseConsoleThunk gpuBaseConsole{{true, outputBuffers.overlayUpdate, overlayUpdater, kit}};

    //----------------------------------------------------------------
    //
    // Module input.
    //
    //----------------------------------------------------------------

    namespace ms = minimalShell;

    ////

    UserPoint userPoint;

    if (inputUpdates.mousePointerUpdate.hasUpdates())
    {
        auto u = inputUpdates.mousePointerUpdate.get();

        if (u.button0)
            (*u.button0 ? userPoint.leftSet : userPoint.leftReset) = true;

        if (u.button1)
            (*u.button1 ? userPoint.rightSet : userPoint.rightReset) = true;

        if (u.position)
            currentMousePos = u.position;
    }

    if (currentMousePos)
    {
        userPoint.valid = true;
        userPoint.floatPos = *currentMousePos;
    }

    ////

    CHECK(currentDisplaySettings->absorb(inputUpdates.displaySettingsUpdate));

    auto desiredOutputSize = currentDisplaySettings->get().desiredOutputSize;

    ////

    GpuMatrix<const uint8_x4> emptyFrame;
    GpuRgbFrameKit gpuRgbFrameKit{emptyFrame};

    //----------------------------------------------------------------
    //
    // Call the test module via minimal shell.
    //
    //----------------------------------------------------------------

    GpuAppAllocKit gpuAppAllocKit = gpuInitKit;

    ////

    auto gpuProcess = [&] (stdPars(auto))
    {
        stdScopedBegin;

        auto& oldKit = kit;
        auto kit = kitCombine(oldKit, GpuRgbFrameKit{emptyFrame});

        ////

        auto processApi = testShell::Process::O | [&] (stdParsNull)
        {
            testModule->process(stdPass);
        };

        testShell->process(processApi, stdPass);

        ////

        stdScopedEnd;
    };

    ////

    auto moduleThunk = ms::engineModuleThunk
    (
        [&] () {return testModule->reallocValid();},
        [&] (stdPars(auto)) {testModule->realloc(stdPassKit(kitCombine(kit, gpuAppAllocKit)));},
        gpuProcess
    );

    ////

    auto msKit = kitCombine
    (
        baseKit,
        ms::BaseImageConsolesKit{&gpuBaseConsole, nullptr, nullptr},
        UserPointKit{userPoint},
        ms::DesiredOutputSizeKit{desiredOutputSize}
    );

    bool sysAllocHappened{};
    minimalShell->process({&args.externalContext, moduleThunk, testModuleMemory, true, sysAllocHappened}, stdPassKit(msKit));

    ////

    stdScopedEnd;
}

//----------------------------------------------------------------

}
