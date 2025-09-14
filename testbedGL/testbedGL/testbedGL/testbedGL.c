#include "testbedGL.h"

#include <thread>

#include "allocation/mallocAllocator/mallocAllocator.h"
#include "allocation/mallocKit.h"
#include "channels/buffers/logBuffer/logBuffer.h"
#include "channels/guiService/guiService.h"
#include "channels/workerService/workerService.h"
#include "checkHeap.h"
#include "configKeeper/configKeeper.h"
#include "debugBeep/debugBeep.h"
#include "errorLog/convertExceptions.h"
#include "errorLog/debugBreak.h"
#include "formattedOutput/formatters/messageFormatterImpl.h"
#include "formattedOutput/userOutputThunks.h"
#include "formatting/messageFormatterKit.h"
#include "gpuLayer/gpuLayerImpl.h"
#include "gpuShell/gpuShell.h"
#include "gui/guiClass.h"
#include "interruptConsole/interruptConsole.h"
#include "lib/logToBuffer/logToBuffer.h"
#include "logKeeper/logKeeper.h"
#include "msgBoxImpl/msgBoxImpl.h"
#include "setThreadName/setThreadName.h"
#include "simpleString/simpleString.h"
#include "stl/stlArray.h"
#include "stlString/stlString.h"
#include "storage/smartPtr.h"
#include "testbedGL/pixelBuffer/pixelBuffer.h"
#include "testbedGL/pixelBufferDrawing/pixelBufferDrawing.h"
#include "testbedGL/testbedGL/setDpiAwareness.h"
#include "testbedGL/windowManager/windowManagerGLFW.h"
#include "threads/threads.h"
#include "timer/timerKit.h"
#include "timerImpl/timerImpl.h"
#include "userOutput/printMsg.h"
#include "userOutput/printMsgEx.h"
#include "worker/worker.h"

namespace testbedGL {

//================================================================
//
// TestModuleFactoryKit
//
//================================================================

KIT_CREATE(TestModuleFactoryKit, const TestModuleFactory&, testModuleFactory);

//================================================================
//
// ExitMessageParams
//
//================================================================

struct ExitMessageParams
{
    size_t maxMessages = 50;
    int printedLevel = msgWarn;
    float32 maxAge = 5.f;
};

KIT_CREATE(ExitMessageParamsKit, ExitMessageParams&, exitMessageParams);

//================================================================
//
// Testbed
//
//================================================================

struct Testbed
{

    //----------------------------------------------------------------
    //
    // Config.
    //
    //----------------------------------------------------------------

    void serialize(const CfgSerializeKit& kit);

    //----------------------------------------------------------------
    //
    // Run.
    //
    //----------------------------------------------------------------

    KIT_CREATE(InterruptKit, InterruptConsole&, interrupt);

    // The GUI thread's internal delta-buffer for global log updates.
    KIT_CREATE2(IntrinsicLogBufferKit, LogBuffer&, intrinsicBuffer, DebuggerOutputControl&, intrinsicBufferDebuggerOutputControl);

    using RunKit = KitCombine<DiagnosticKit, InterruptKit, IntrinsicLogBufferKit, TimerKit, MallocKit, TestModuleFactoryKit, ExitMessageParamsKit>;

    void run(stdPars(RunKit));

    //----------------------------------------------------------------
    //
    // RunExKit
    //
    //----------------------------------------------------------------

    using GpuBasicKit = KitCombine<GpuInitKit, GpuPropertiesKit, GpuCurrentContextKit, GpuExecKit, GpuCurrentStreamKit>;

    ////

    KIT_CREATE(GuiClassKit, GuiClass&, guiClass);

    ////

    using RunExKit = KitCombine<RunKit, GpuBasicKit, GuiClassKit>;

    //----------------------------------------------------------------
    //
    // WindowReinitRequestKit
    //
    //----------------------------------------------------------------

    struct WindowReinitRequest
    {
        bool on = false;
    };

    KIT_CREATE(WindowReinitRequestKit, WindowReinitRequest&, windowReinitRequest);

    //----------------------------------------------------------------
    //
    // Window reinit cycle.
    //
    //----------------------------------------------------------------

    KIT_CREATE(WindowManagerKit, WindowManager&, windowManager);

    ////

    KIT_CREATE(GuiServiceKit, guiService::ServerApi&, guiService);
    KIT_CREATE(WorkerServiceKit, workerService::ClientApi&, workerService);
    KIT_CREATE(ConfigServiceKit, configService::ClientApi&, configService);
    KIT_CREATE(LogServiceKit, logService::ClientApi&, logService);

    using ServicesKit = KitCombine<GuiServiceKit, WorkerServiceKit, ConfigServiceKit, LogServiceKit>;

    ////

    KIT_CREATE(GuiSerializationKit, CfgSerialization&, guiSerialization);

    ////

    using WindowReinitCycleKit = KitCombine<RunExKit, WindowManagerKit, WindowReinitRequestKit, ServicesKit, GuiSerializationKit>;

    void windowReinitCycle(stdPars(WindowReinitCycleKit));

    //----------------------------------------------------------------
    //
    // Event loop.
    //
    //----------------------------------------------------------------

    KIT_CREATE(MainWindowKit, Window&, mainWindow);
    KIT_CREATE(BufferDrawingKit, PixelBufferDrawing&, pixelBufferDrawing);
    KIT_CREATE(PixelBufferKit, PixelBuffer&, pixelBuffer);

    using EventLoopKit = KitCombine<RunExKit, MainWindowKit, WindowManagerKit, WindowReinitRequestKit,
        BufferDrawingKit, PixelBufferKit, ServicesKit, GuiSerializationKit>;

    void eventLoop(stdPars(EventLoopKit));

    //----------------------------------------------------------------
    //
    // Gpu config.
    //
    //----------------------------------------------------------------

    GpuContextHelper gpuContextHelper;

    //----------------------------------------------------------------
    //
    // Display config.
    //
    //----------------------------------------------------------------

    struct DisplayConfig
    {
        bool steady = true;

        MultiSwitch<WindowMode, WindowMode::COUNT, WindowMode::Maximized> mode;

        BoolSwitch verticalSync{true};

        BoolSwitch decorated{false};
        StandardSignal maxScreenSignal;
        StandardSignal fullScreenSignal;

        NumericVar<Point<Space>> pos{point(-0x7FFF), point(+0x7FFF), point(0)};
        NumericVar<Point<Space>> size{point(0), point(+0x7FFF), point(1280, 720)};
    };

    DisplayConfig display;

    //----------------------------------------------------------------
    //
    // ExitMessageCfg
    //
    //----------------------------------------------------------------

    struct ExitMessageCfg
    {
        ExitMessageParams defaultParams;

        NumericVar<size_t> maxMessages{0, typeMax<size_t>(), defaultParams.maxMessages};
        NumericVar<int> printedLevel{0, typeMax<int>(), defaultParams.printedLevel};
        NumericVar<float32> maxAge{0, typeMax<float32>(), defaultParams.maxAge};
    };

    ExitMessageCfg exitMessageCfg;

};

//================================================================
//
// Testbed::serialize
//
//================================================================

void Testbed::serialize(const CfgSerializeKit& kit)
{
    {
        CFG_NAMESPACE("GUI");

        {
            CFG_NAMESPACE("GPU Init");

            gpuContextHelper.serialize(kit);
        }

        ////

        {
            CFG_NAMESPACE("Display");

            check_flag(display.verticalSync.serialize(kit, STR("Vertical Sync"), STR("F2"), STR("Toggle vertical sync")), display.steady);

            bool modeSteady = display.mode.serialize
            (
                kit, STR("Window Mode"),
                STR("Minimized"),
                STR("Normal"),
                STR("Maximized"),
                STR("Fullscreen")
            );

            check_flag(modeSteady, display.steady);

            check_flag(display.decorated.serialize(kit, STR("Window Decorated"), STR("Alt+F11"), STR("Toggle window title display")), display.steady);
            check_flag(display.pos.serialize(kit, STR("Window Pos")), display.steady);
            check_flag(display.size.serialize(kit, STR("Window Size")), display.steady);

            if_not (display.fullScreenSignal.serialize(kit, STR("Fullscreen"), STR("Alt+Enter"), STR("Toggle fullscreen mode")))
            {
                if (display.mode != WindowMode::FullScreen)
                    display.mode = WindowMode::FullScreen;
                else
                    {display.mode = WindowMode::Normal; display.decorated = true;}

                display.steady = false;
                display.fullScreenSignal.clear();
            }

            if_not (display.maxScreenSignal.serialize(kit, STR("MaxScreen"), STR("F11"), STR("Toggle maxscreen mode (maximized window without a title)")))
            {
                if (display.mode == WindowMode::Maximized && !display.decorated)
                    {display.mode = WindowMode::Normal; display.decorated = true;}
                else
                    {display.mode = WindowMode::Maximized; display.decorated = false;}

                display.steady = false;
                display.maxScreenSignal.clear();
            }
        }

        ////

        {
            CFG_NAMESPACE("Exit Messages");

            exitMessageCfg.maxAge.serialize(kit, STR("Max Age In Seconds"));
            exitMessageCfg.printedLevel.serialize(kit, STR("Printed Level"), STR("0 info, 1 warnings, 2 errors, 3 nothing"));
            exitMessageCfg.maxMessages.serialize(kit, STR("Max Messages"));
        }
    }
}

//================================================================
//
// Testbed::run
//
//================================================================

void Testbed::run(stdPars(RunKit))
{
    stdExceptBegin;

    //----------------------------------------------------------------
    //
    // Disabling any OS-side scaling of the window content.
    //
    //----------------------------------------------------------------

    setDpiAwareness(stdPass);

    //----------------------------------------------------------------
    //
    // Create all subsystems.
    //
    //----------------------------------------------------------------

    UniqueInstance<GuiClass> guiClass;

    UniquePtr<Worker> worker = Worker::create(kit.testModuleFactory.create());

    UniqueInstance<ConfigKeeper> configKeeper;

    UniqueInstance<LogKeeper> logKeeper;

    //----------------------------------------------------------------
    //
    // * Init the config keeper.
    //
    // * Load and update the config file for all subsystems
    // while they are in stopped state.
    //
    //----------------------------------------------------------------

    {
        auto initialSerialization = cfgSerializationThunk | [&] (auto& kit)
        {
            this->serialize(kit);
            guiClass->serialize(kit);
            configKeeper->serialize(kit);
            logKeeper->serialize(kit);
            worker->serialize(kit);
        };

        SimpleString baseName; baseName << kit.testModuleFactory.configName() << CT(".json");
        REQUIRE(def(baseName));

        configKeeper->init({baseName.cstr(), initialSerialization}, stdPass);
    }

    REMEMBER_CLEANUP(configKeeper.reset());

    //----------------------------------------------------------------
    //
    // Exit messages support:
    //
    // * Update the exit messages config.
    //
    // * At the end, take the global log from GUI class,
    // append the intrinsic delta-buffer and return the result.
    //
    //----------------------------------------------------------------

    kit.exitMessageParams.maxAge = exitMessageCfg.maxAge;
    kit.exitMessageParams.printedLevel = exitMessageCfg.printedLevel;
    kit.exitMessageParams.maxMessages = exitMessageCfg.maxMessages;

    ////

    UniqueInstance<LogBuffer> exitLog;

    auto exitLogHandling = [&] ()
    {
        exitLog->absorb(kit.intrinsicBuffer);
        kit.intrinsicBuffer.moveFrom(*exitLog);
    };

    REMEMBER_CLEANUP(exitLogHandling());

    //----------------------------------------------------------------
    //
    // Create all boards.
    //
    //----------------------------------------------------------------

    UniquePtr<guiService::Board> guiService;
    convertExceptions(guiService = guiService::Board::create());

    UniquePtr<workerService::Board> workerService;
    convertExceptions(workerService = workerService::Board::create());

    UniquePtr<configService::Board> configService;
    convertExceptions(configService = configService::Board::create());

    UniquePtr<logService::Board> logService;
    convertExceptions(logService = logService::Board::create());

    //----------------------------------------------------------------
    //
    // * Init the log keeper.
    //
    // * Start the log keeper thread; if successful,
    // remember to shutdown the thread in any case.
    //
    //----------------------------------------------------------------

    {
        SimpleString baseName; baseName << kit.testModuleFactory.configName() << CT(".log");
        REQUIRE(def(baseName));

        logKeeper->init({baseName.cstr()}, stdPass);
    }

    REMEMBER_CLEANUP(logKeeper.reset());

    ////

    UniqueInstance<ShutdownBuffer> logKeeperShutdownRequest;
    logKeeperShutdownRequest->addRequest();

    ////

    auto logKeeperArgs = LogKeeper::RunArgs{*logService, *guiService};

    std::thread logKeeperThread;
    convertExceptions(logKeeperThread = std::thread(&LogKeeper::run, logKeeper.get(), logKeeperArgs));

    ////

    auto logKeeperFinalUpdate = [&] ()
    {
        stdExceptBegin;

        UniqueInstance<TextBuffer> update;

        auto receiver = LogBufferReceiver::O | [&] (auto& text, auto& kind, auto& moment)
            {update->addLine(text.ptr, text.size);}; // may throw

        kit.intrinsicBuffer.readLastMessages(receiver, typeMax<size_t>());

        REQUIRE(logService->addTextUpdate(*update));

        stdExceptEnd;
    };

    ////

    auto shutdownLogKeeper = [&] ()
    {
        errorBlock(logKeeperFinalUpdate());

        logService->addShutdownRequest(*logKeeperShutdownRequest);
        logKeeperThread.join();
    };

    ////

    REMEMBER_CLEANUP(shutdownLogKeeper());

    //----------------------------------------------------------------
    //
    // Start the config keeper; if successful,
    // remember to shutdown the thread in any case.
    //
    //----------------------------------------------------------------

    UniqueInstance<ShutdownBuffer> configKeeperShutdownRequest;
    configKeeperShutdownRequest->addRequest();

    ////

    auto configKeeperArgs = ConfigKeeper::RunArgs{*configService, *guiService};

    std::thread configKeeperThread;
    convertExceptions(configKeeperThread = std::thread(&ConfigKeeper::run, configKeeper.get(), configKeeperArgs));

    ////

    auto shutdownConfigKeeper = [&] ()
    {
        configService->addShutdownRequest(*configKeeperShutdownRequest);
        configKeeperThread.join();
    };

    REMEMBER_CLEANUP(shutdownConfigKeeper());

    //----------------------------------------------------------------
    //
    // Create GPU compute context and stream.
    //
    //----------------------------------------------------------------

    //
    // Init API.
    //

    GpuInitApiImpl gpuInitApi(kit);
    gpuInitApi.initialize(stdPass);
    auto gpuInitKit = gpuInitApi.getKit();

    //
    // Properties.
    //

    GpuProperties gpuProperties;
    GpuContextOwner gpuContext;
    gpuContextHelper.createContext(gpuProperties, gpuContext, stdPassKit(kitCombine(kit, gpuInitKit)));

    //
    // Streams.
    //

    GpuStreamOwner guiStream;
    gpuInitKit.gpuStreamCreation.createStream(gpuContext, false, guiStream, stdPass);

    ////

    GpuStreamOwner workerStream;
    gpuInitKit.gpuStreamCreation.createStream(gpuContext, false, workerStream, stdPass);

    //
    // Profiler: Not implemented.
    //

    ProfilerKit profilerNullKit{nullptr};

    //
    // Exec API.
    //

    GpuExecApiImpl gpuExecApi{kitCombine(kit, profilerNullKit)};
    GpuExecKit gpuExecKit = gpuExecApi.getKit();

    ////

    auto& oldKit = kit;

    auto kit = kitCombine
    (
        oldKit,
        gpuInitKit,
        gpuExecKit,
        GpuPropertiesKit{gpuProperties},
        GpuCurrentContextKit{gpuContext},
        GpuCurrentStreamKit{guiStream}
    );

    //----------------------------------------------------------------
    //
    // Deallocate UI board GPU buffers before GPU context deinit
    // (it's because I want to create the boards very early).
    //
    //----------------------------------------------------------------

    UniqueInstance<OverlayBuffer> tmpOverlayBuffer;

    REMEMBER_CLEANUP(guiService->takeOverlayUpdate(*tmpOverlayBuffer));

    //----------------------------------------------------------------
    //
    // Set slightly higher priority for the UI thread.
    //
    //----------------------------------------------------------------

    setThreadName(STR("~GUI"));

    ////

#if defined(_WIN32)
    ThreadControl currentThread;
    threadGetCurrent(currentThread, stdPass);

    REQUIRE(currentThread->setPriority(ThreadPriorityPlus1));
#endif

    //----------------------------------------------------------------
    //
    // GLFW init.
    //
    //----------------------------------------------------------------

    WindowManagerGLFW windowManager;

    windowManager.init(stdPass);

    //----------------------------------------------------------------
    //
    // Init / deinit GUI.
    //
    // Before deinit, take the global log from it.
    //
    //----------------------------------------------------------------

    Point<Space> currentResolution{};
    windowManager.getCurrentDisplayResolution(currentResolution, stdPass);

    ////

    auto guiSerialization = cfgSerializationThunk | [&] (auto& kit)
    {
        this->serialize(kit);
        guiClass->serialize(kit);
    };

    guiClass->init({currentResolution, guiSerialization}, stdPass);

    REMEMBER_CLEANUP(guiClass.reset());

    ////

    REMEMBER_CLEANUP(guiClass->takeGlobalLog(*exitLog));

    //----------------------------------------------------------------
    //
    // Create GUI signaller.
    //
    //----------------------------------------------------------------

    auto guiSignaller = guiService::Signaller::O | [&] ()
    {
        // Post an empty event to the event queue.
        windowManager.postEmptyEvent();
    };

    ////

    guiService->setSignaller(&guiSignaller);
    REMEMBER_CLEANUP(guiService->setSignaller(nullptr));

    ////

    kit.interrupt.setSignaller(&guiSignaller);
    REMEMBER_CLEANUP(kit.interrupt.setSignaller(nullptr));

    //----------------------------------------------------------------
    //
    // Init WORKER.
    // Start WORKER. If successful, remember to shutdown the thread in any case.
    //
    //----------------------------------------------------------------

    UniquePtr<ContextBinder> workerGLContext;
    windowManager.createOffscreenGLContext(workerGLContext, stdPass);

    ////

    workerGLContext->bind(stdPass);
    REQUIRE(glewInit() == GLEW_OK);
    workerGLContext->unbind(stdPass);

    ////

    auto externalsFactory = worker::ExternalsFactory::O | [&] ()
    {
        return kit.testModuleFactory.createExternals();
    };

    ////

    worker->init({*workerGLContext, externalsFactory, *guiService}, stdPass);

    auto workerDeinit = [&] (stdPars(auto))
    {
        workerGLContext->bind(stdPass);
        worker.reset();
        workerGLContext->unbind(stdPass);
    };

    REMEMBER_CLEANUP(errorBlock(workerDeinit(stdPassNc)));

    ////

    auto externalContext = minimalShell::GpuExternalContext
    {
        gpuProperties,
        gpuContext,
        workerStream
    };

    ////

    UniqueInstance<ShutdownBuffer> workerShutdownRequest;
    workerShutdownRequest->addRequest();

    ////

    auto workerArgs = Worker::RunArgs{externalContext, *workerService, *guiService, *configService};

    std::thread workerThread;
    convertExceptions(workerThread = std::thread(&Worker::run, worker.get(), workerArgs));

    ////

    auto workerShutdown = [&] ()
    {
        workerService->addShutdownRequest(*workerShutdownRequest);
        workerThread.join();
    };

    REMEMBER_CLEANUP(workerShutdown());

    ////

    REMEMBER_CLEANUP(errorBlock(kit.gpuStreamWaiting.waitStream(workerStream, stdPassNc)));

    //----------------------------------------------------------------
    //
    // Window re-creation loop.
    //
    //----------------------------------------------------------------

    for (; ;)
    {
        WindowReinitRequest windowReinitRequest;

        auto kitEx = kitCombine
        (
            kit,
            WindowManagerKit{windowManager},
            WindowReinitRequestKit{windowReinitRequest},
            GuiClassKit{*guiClass},
            GuiServiceKit{*guiService},
            WorkerServiceKit{*workerService},
            ConfigServiceKit{*configService},
            LogServiceKit{*logService},
            GuiSerializationKit{guiSerialization}
        );

        windowReinitCycle(stdPassKit(kitEx));

        if_not (windowReinitRequest.on)
            break;
    }

    ////

    stdExceptEnd;
}

//================================================================
//
// Testbed::windowReinitCycle
//
//================================================================

void Testbed::windowReinitCycle(stdPars(WindowReinitCycleKit))
{

    //----------------------------------------------------------------
    //
    // Create a window.
    //
    //----------------------------------------------------------------

    WindowCreationArgs wc;

    wc.name = kit.testModuleFactory.displayName();
    wc.location.mode = WindowMode(display.mode());
    wc.location.verticalSync = display.verticalSync;
    wc.location.decorated = display.decorated;
    wc.location.pos = display.pos;
    wc.location.size = display.size;
    wc.resizable = true;

    UniquePtr<Window> mainWindow;
    kit.windowManager.createWindow(mainWindow, wc, stdPass);

    //----------------------------------------------------------------
    //
    // Color buffer drawer.
    //
    //----------------------------------------------------------------

    UniqueInstance<PixelBufferDrawing> pixelBufferDrawing;

    pixelBufferDrawing->reinit(stdPass);

    //----------------------------------------------------------------
    //
    // Create shared GL <-> CUDA buffer.
    //
    //----------------------------------------------------------------

    PixelBuffer pixelBuffer;

    Point<Space> imageSize{};
    mainWindow->getImageSize(imageSize, stdPass);

    pixelBuffer.realloc<uint8_x4>(imageSize, kit.gpuProperties.samplerRowAlignment, stdPass);

    //----------------------------------------------------------------
    //
    // Message loop.
    //
    //----------------------------------------------------------------

    {
        auto kitEx = kitCombine
        (
            kit,
            MainWindowKit{*mainWindow},
            BufferDrawingKit{*pixelBufferDrawing},
            PixelBufferKit{pixelBuffer}
        );

        eventLoop(stdPassKit(kitEx));
    }
}

//================================================================
//
// Testbed::eventLoop
//
//================================================================

void Testbed::eventLoop(stdPars(EventLoopKit))
{
    stdScopedBegin;

    Window& mainWindow = kit.mainWindow;

    //----------------------------------------------------------------
    //
    // Draw receiver.
    //
    //----------------------------------------------------------------

    auto drawReceiverFunc = [&] (auto& drawer, stdParsNull)
    {

        Point<Space> imageSize{};
        mainWindow.getImageSize(imageSize, stdPass);

        //
        // Resize the pixel buffer if neccessary.
        //

        REQUIRE(imageSize >= 0);

        PixelBuffer& pixelBuffer = kit.pixelBuffer;

        if_not (imageSize == pixelBuffer.size())
            pixelBuffer.realloc<uint8_x4>(imageSize, kit.gpuProperties.samplerRowAlignment, stdPass);

        //
        // Fill the pixel buffer with CUDA.
        //

        {
            auto baseStream = getNativeHandle(kit.gpuCurrentStream);

            pixelBuffer.lock(baseStream, stdPass);
            REMEMBER_CLEANUP_EX(lockCleanup, errorBlock(pixelBuffer.unlock(baseStream, stdPassNc)));

            ////

            GpuMatrix<uint8_x4> tmp;
            pixelBuffer.getComputeBuffer(tmp, stdPass);

            drawer(tmp, stdPass);

            ////

            lockCleanup.cancel();
            pixelBuffer.unlock(baseStream, stdPass);
        }

        //
        // Draw the pixel buffer.
        //

        kit.pixelBufferDrawing.draw(pixelBuffer, point(0), stdPass);

        //
        // Swap buffers.
        //

        mainWindow.swapBuffers(stdPass);
    };

    ////

    auto drawReceiver = gui::DrawReceiver::O | drawReceiverFunc;

    //----------------------------------------------------------------
    //
    // Event processing loop.
    //
    // Inside the loop, error recovery is performed.
    //
    //----------------------------------------------------------------

    mainWindow.setVisible(true, stdPass);

    ////

    for (; ;)
    {
        bool verticalSyncSave = display.verticalSync;

        //----------------------------------------------------------------
        //
        // Process events.
        //
        //----------------------------------------------------------------

        auto eventSource = gui::EventSource::O | [&] (bool waitEvents, auto& waitTimeoutMs, auto& receivers, stdParsNull)
        {
            mainWindow.getEvents(waitEvents, waitTimeoutMs, receivers, stdPassThru);
        };

        ////

        auto shutdownRequest = gui::ShutdownRequest{};

        GuiClass::ProcessArgs guiArgs
        {
            kit.intrinsicBuffer,
            kit.intrinsicBufferDebuggerOutputControl,
            eventSource,
            drawReceiver,
            shutdownRequest,
            kit.guiService,
            kit.workerService,
            kit.configService,
            kit.logService,
            kit.guiSerialization
        };

        errorBlock(kit.guiClass.processEvents(guiArgs, stdPassNc));

        //----------------------------------------------------------------
        //
        // Exit checking.
        //
        //----------------------------------------------------------------

        if_not (errorBlock(mainWindow.shouldContinue(stdPassNc)))
            break;

        if (shutdownRequest.on)
            break;

        if (kit.interrupt())
        {
            printMsg(kit.msgLog, STR("Exiting due to interrupt signal..."));
            break;
        }

        //----------------------------------------------------------------
        //
        // Window location.
        //
        //----------------------------------------------------------------

        auto handleWindowLocation = [&] ()
        {
            if_not (display.steady)
            {
                display.steady = true;

                WindowLocation location;
                location.mode = WindowMode(display.mode());
                location.verticalSync = display.verticalSync;
                location.decorated = display.decorated;
                location.pos = display.pos;
                location.size = display.size;

                mainWindow.setWindowLocation(location, stdPass);

                if_not (verticalSyncSave == display.verticalSync)
                    printMsg(kit.msgLog, STR("Vertical sync %"), display.verticalSync());
            }
            else
            {
                WindowLocation location;
                mainWindow.getWindowLocation(location, stdPass);

                display.mode = location.mode;
                display.decorated = location.decorated;

                if (location.mode == WindowMode::Normal)
                {
                    display.pos = location.pos;
                    display.size = location.size;
                }
            }
        };

        errorBlock(handleWindowLocation());

    }

    stdScopedEnd;
}

//================================================================
//
// mainEntry
//
// Create basic kits, create LogBufferStl for global log, catch exceptions,
// output last 5 sec of global log messages to msg box on exit.
//
//================================================================

bool mainEntry(int argCount, const CharType* argStr[], const TestModuleFactory& factory)
{
    MsgBoxImpl msgBox;

    bool ok = false;

    try
    {

        try
        {

            InterruptConsole interruptConsole;

            //----------------------------------------------------------------
            //
            // Msg log
            //
            //----------------------------------------------------------------

            TimerImpl timer;

            ////

            StlArray<CharType> formatterArray;

            if_not (formatterArray.reallocBool(65536))
                throw std::bad_alloc();

            MessageFormatterImpl formatter{formatterArray};

            ////

            // The GUI thread's internal delta-buffer for global log updates.
            UniqueInstance<LogBuffer> intrinsicBuffer;

            ////

            auto dummyUpdater = LogUpdater::O | [&] () {};

            LogToBufferThunk msgLog{{*intrinsicBuffer, formatter, timer, typeMax<int>(), dummyUpdater}};

            ////

            ErrorLogByMsgLog errorLog(msgLog);
            ErrorLogKit errorLogKit(errorLog);

            MsgLogExByMsgLog msgLogEx(msgLog);

            ////

            MAKE_MALLOC_ALLOCATOR(errorLogKit);

            ////

            ExitMessageParams exitMessageCfg;

            ////

            auto kit = kitCombine
            (
                Testbed::InterruptKit{interruptConsole},
                TimerKit{timer},
                MessageFormatterKit{formatter},
                MsgLogKit{msgLog},
                ErrorLogKit{errorLog},
                MsgLogExKit{msgLogEx},
                MallocKit{mallocAllocator},
                Testbed::IntrinsicLogBufferKit{*intrinsicBuffer, msgLog},
                TestModuleFactoryKit{factory},
                ExitMessageParamsKit{exitMessageCfg}
            );

            ////

            stdTraceRoot;

            //----------------------------------------------------------------
            //
            // Continue
            //
            //----------------------------------------------------------------

            Testbed testshell;

            ok = errorBlock(testshell.run(stdPassNc));

            //----------------------------------------------------------------
            //
            // Display last log messages.
            //
            //----------------------------------------------------------------

            {
                auto currentMoment = timer.moment();
                MsgKind outputKind = msgInfo;
                SimpleString outputStr;

                ////

                auto receiver = LogBufferReceiver::O | [&] (const CharArray& text, MsgKind kind, const TimeMoment& moment)
                {
                    if_not (kind >= exitMessageCfg.printedLevel)
                        return;

                    if_not (timer.diff(moment, currentMoment) <= exitMessageCfg.maxAge)
                        return;

                    if (kind > outputKind)
                        outputKind = kind;

                    if (outputStr.size())
                        outputStr += STR("\n");

                    outputStr += text;
                };

                ////

                intrinsicBuffer->readLastMessages(receiver, exitMessageCfg.maxMessages);

                if_not (outputStr.valid())
                    throw std::bad_alloc();

                if (outputStr.size())
                    ensure(msgBox(outputStr.cstr(), outputKind));
            }

        }
        catch (const std::exception& e)
        {
            auto s = CT("STL exception: ") + StlString(e.what());
            msgBox(s.c_str(), msgErr);
        }
        catch (const CharType* msg)
        {
            auto s = CT("Hard exception: ") + StlString(msg);
            msgBox(s.c_str(), msgErr);
        }
    }
    catch (const std::bad_alloc&)
    {
        msgBox(CT("Insufficient memory"), msgErr);
    }

    //----------------------------------------------------------------
    //
    // Check heap
    //
    //----------------------------------------------------------------

    if (!checkHeapIntegrity())
        {msgBox(CT("HEAP MEMORY IS DAMAGED!"), msgErr); return false;}
    else if (!checkHeapLeaks())
        {msgBox(CT("MEMORY LEAKS ARE DETECTED!"), msgErr); return false;}

    ////

    return ok;
}

//----------------------------------------------------------------

}
