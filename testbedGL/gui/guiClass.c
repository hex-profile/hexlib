#include "guiClass.h"

#include "cfgTools/boolSwitch.h"
#include "cfgTools/multiSwitch.h"
#include "channels/buffers/textBuffer/textBuffer.h"
#include "compileTools/blockExceptionsSilent.h"
#include "cfgVars/cfgSerializeImpl/cfgSerializeImpl.h"
#include "cfgTools/cfgSimpleString.h"
#include "dataAlloc/memoryAllocatorStubs.h"
#include "errorLog/convertExceptions.h"
#include "errorLog/debugBreak.h"
#include "formattedOutput/requireMsg.h"
#include "gui/drawErrorPattern/drawErrorPattern.h"
#include "gui/guiModule.h"
#include "keyMap.h"
#include "kits/userPoint.h"
#include "lib/keyBuffer/keyBuffer.h"
#include "lib/signalSupport/signalSupport.h"
#include "minimalShell/minimalShell.h"
#include "signalsTools/signalTools.h"
#include "storage/rememberCleanup.h"
#include "userOutput/msgLog.h"
#include "userOutput/printMsgEx.h"

namespace gui {

using minimalShell::MinimalShell;

//================================================================
//
// GuiClassImpl
//
//================================================================

struct GuiClassImpl : public GuiClass
{

    //----------------------------------------------------------------
    //
    // Config.
    //
    //----------------------------------------------------------------

    void serialize(const CfgSerializeKit& kit);

    //----------------------------------------------------------------
    //
    // Init.
    //
    //----------------------------------------------------------------

    stdbool init(const InitArgs& args, stdPars(InitKit));

    ////

    virtual void takeGlobalLog(LogBuffer& result)
    {
        result.moveFrom(*globalLogBuffer);
    }

    //----------------------------------------------------------------
    //
    // Process.
    //
    //----------------------------------------------------------------

    stdbool processEvents(const ProcessArgs& args, stdPars(ProcessKit));

    //----------------------------------------------------------------
    //
    // processInputUpdates
    //
    //----------------------------------------------------------------

    struct ProcessInputUpdatesArgs
    {
        guiService::ServerApi& guiService;
        workerService::ClientApi& workerService;
        logService::ClientApi& logService;
        LogBuffer& intrinsicBuffer;
        CfgSerialization& guiSerialization;
    };

    stdbool processInputUpdates(const ProcessInputUpdatesArgs& args, stdPars(ProcessKit));

    //----------------------------------------------------------------
    //
    // Update-and-redraw.
    //
    // Processes input updates and redraws the image.
    //
    // The input updates are checked on every redraw because:
    //
    // * It is fast if there are no real updates (to take a mutex).
    //
    // * The refresh handler may be called multiple times within "get events",
    //   for example, when resizing the window. If the WORKER wants to update
    //   at the same time, let it update.
    //
    //----------------------------------------------------------------

    using DrawArgs = GuiModule::DrawArgs;

    ////

    stdbool drawNormal(const DrawArgs& args, stdPars(ProcessKit));

    stdbool drawErrorPattern(const DrawArgs& args, stdPars(ProcessKit));

    ////

    struct UpdateAndRedrawArgs
    {
        guiService::ServerApi& guiService;
        workerService::ClientApi& workerService;
        logService::ClientApi& logService;
        LogBuffer& intrinsicBuffer;
        CfgSerialization& guiSerialization;
        const GpuMatrix<uint8_x4>& dstImage;
    };

    stdbool updateAndRedraw(const UpdateAndRedrawArgs& args, stdPars(ProcessKit));

    //----------------------------------------------------------------
    //
    // printHelp
    //
    //----------------------------------------------------------------

    stdbool printHelp(const ProcessArgs& args, bool printAlgoKeys, stdPars(ProcessKit));

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

    bool initialized = false;

    //----------------------------------------------------------------
    //
    // GUI module support and memory: the order of construction is important.
    //
    //----------------------------------------------------------------

    UniqueInstance<MinimalShell> minimalShell;

    void minimalShellInit()
    {
        auto& s = minimalShell->settings();

        s.setGpuContextMaintainer(false);
        s.setProfilerShellHotkeys(false);
        s.setGpuShellHotkeys(false);
        s.setDisplayParamsHotkeys(false);
        s.setBmpConsoleHotkeys(false);
    }

    int minimalShellInitCaller = (minimalShellInit(), 0);

    ////

    MemController guiModuleMemory;

    UniqueInstance<GuiModule> guiModule;

    //----------------------------------------------------------------
    //
    // GUI internal key / signal support.
    //
    //----------------------------------------------------------------

    SignalSupport signalSupport;

    UniqueInstance<KeyBuffer> keyBuffer;

    //----------------------------------------------------------------
    //
    // Config.
    //
    //----------------------------------------------------------------

    StandardSignal exitSignal;

    ////

    BoolVar welcomeMessage{true};
    StandardSignal welcomeMessageToggle;
    StandardSignal printHelpSignal;
    StandardSignal printAlgoKeysSignal;

    ////

    NumericVar<size_t> gLogHistoryLimit{0, typeMax<size_t>(), 10000};
    NumericVar<int> gLogDebuggerOutputLevel{0, typeMax<int>(), msgErr};

    ////

    enum class WorkerByMouse {None, Press, PressAndRelease, All, COUNT};

    MultiSwitch<WorkerByMouse, WorkerByMouse::COUNT, WorkerByMouse::Press> workerByMouse;

    //----------------------------------------------------------------
    //
    // Input buffers.
    //
    //----------------------------------------------------------------

    workerService::BufferInstances updatesToWorkerMemory;
    workerService::BufferRefs updatesToWorker = updatesToWorkerMemory.refs;

    ////

    OptionalObject<Point<Space>> desiredOutputSize;
    mousePointerBuffer::MousePointer mousePointer;

    //----------------------------------------------------------------
    //
    // Input buffers and persistent buffers.
    //
    //----------------------------------------------------------------

    // Input buffers.
    UniqueInstance<ShutdownBuffer> shutdownUpdate;
    UniqueInstance<LogBuffer> globalLogUpdate;
    UniqueInstance<LogBuffer> localLogUpdate;
    UniqueInstance<LogBuffer> specialLogUpdate;
    UniqueInstance<ActionSetBuffer> actionSetUpdate;
    UniqueInstance<CfgTree> configInputUpdate;

    // Persistent buffers.
    UniqueInstance<ShutdownBuffer> shutdownBuffer;
    UniqueInstance<OverlayBuffer> overlayBuffer;
    UniqueInstance<LogBuffer> globalLogBuffer;
    UniqueInstance<LogBuffer> localLogBuffer;
    UniqueInstance<ActionSetBuffer> actionSet;

    //----------------------------------------------------------------
    //
    // Worker key map.
    //
    //----------------------------------------------------------------

    KeyMap workerKeyMap;

    //----------------------------------------------------------------
    //
    // Editor.
    //
    //----------------------------------------------------------------

    static auto getBasicTextEditor()
    {
        #if defined(_WIN32)
            return STR("notepad");
        #elif defined(__linux__)
            return STR("gedit");
        #else
            #error
        #endif
    }

    static CharArray getDefaultTextEditor()
    {
        auto env = getenv(CT("HEXLIB_TEXT_EDITOR"));
        return env ? charArrayFromPtr(env) : getBasicTextEditor();
    }

    SimpleStringVar textEditor{getDefaultTextEditor()};
    UniqueInstance<EditRequest> editRequest;

    auto textEditorHelpMessage()
    {
        return STR("Please set a valid editor in the \"Text Editor\" parameter of the config file.");
    }

    //----------------------------------------------------------------
    //
    // Config updating and editing support.
    //
    //----------------------------------------------------------------

    UniqueInstance<CfgTree> configOutputBuffer;

    cfgVarsImpl::CfgTemporary configTemp;

    StandardSignal editConfig;

    //----------------------------------------------------------------
    //
    // Log keeper usage.
    //
    //----------------------------------------------------------------

    UniqueInstance<TextBuffer> logKeeperUpdate;

    StandardSignal editLog;

};

//----------------------------------------------------------------

UniquePtr<GuiClass> GuiClass::create()
    {return makeUnique<GuiClassImpl>();}

//================================================================
//
// GuiClassImpl::serialize
//
//================================================================

void GuiClassImpl::serialize(const CfgSerializeKit& kit)
{
    {
        CFG_NAMESPACE("GUI");

        {
            CFG_NAMESPACE("Text Logs");

            {
                CFG_NAMESPACE("Global Log");

                if_not (gLogHistoryLimit.serialize(kit, STR("History Limit")))
                    globalLogBuffer->setHistoryLimit(gLogHistoryLimit);

                gLogDebuggerOutputLevel.serialize(kit, STR("Debugger Output Level"), STR("0 info, 1 warnings, 2 errors, 3 nothing"));
            }
        }

        guiModule->serialize(kit);

        ////

        exitSignal.serialize(kit, STR("Exit"), STR("F10"), STR("Exit the program"));

        welcomeMessage.serialize(kit, STR("Welcome Message"));
        welcomeMessageToggle.serialize(kit, STR("Toggle welcome message"), STR("Ctrl+F1"), STR("Toggle welcome message"));
        printHelpSignal.serialize(kit, STR("Print help"), STR("F1"), STR("Print help"));
        printAlgoKeysSignal.serialize(kit, STR("Print algorithm module keys"), STR("Alt+F1"), STR("Print algorithm module keys"));

        ////

        workerByMouse.serialize
        (
            kit, STR("Worker By Mouse"),
            STR("None"),
            STR("Press"),
            STR("And Release"),
            STR("All")
        );

        textEditor.serialize(kit, STR("Text Editor"));
        editConfig.serialize(kit, STR("Edit Config"), STR("`"), STR("Edit config: Press Accent/Tilde key"));
        editLog.serialize(kit, STR("Edit Log"), STR("Shift+`"), STR("Edit log: Press Shift+Accent/Tilde key"));
    }

    ////

    {
        CFG_NAMESPACE("GUI Drawer Shell");
        minimalShell->serialize(kit);

        {
            CFG_NAMESPACE("GUI Module Memory");
            guiModuleMemory.serialize(kit);
        }
    }
}

//================================================================
//
// GuiClassImpl::init
//
//================================================================

stdbool GuiClassImpl::init(const InitArgs& args, stdPars(InitKit))
{
    initialized = false;

    ////

    require(minimalShell->init(stdPass));

    ////

    require(keyBuffer->reserve(64, stdPass));

    require(signalSupport.initSignals(args.guiSerialization, stdPass));

    ////

    if (welcomeMessage)
        printMsg(kit.msgLog, STR("Welcome. Press F1 for help. Press Ctrl+F1 to disable this message."));

    ////

    initialized = true;

    returnTrue;
}

//================================================================
//
// GuiClassImpl::drawErrorPattern
//
//================================================================

stdbool GuiClassImpl::drawErrorPattern(const DrawArgs& args, stdPars(ProcessKit))
{
    stdScopedBegin;

    //----------------------------------------------------------------
    //
    // Extend kit.
    //
    //----------------------------------------------------------------

    auto& oldKit = kit;

    AllocatorForbidden<CpuAddrU> cpuAllocator{kit};
    AllocatorForbidden<GpuAddrU> gpuAllocator{kit};

    auto kit = kitCombine
    (
        oldKit,
        ProfilerKit{nullptr},
        CpuFastAllocKit{cpuAllocator},
        GpuFastAllocKit{gpuAllocator},
        DataProcessingKit{true}
    );

    //----------------------------------------------------------------
    //
    // Draw.
    //
    //----------------------------------------------------------------

    require(::drawErrorPattern(args.dstImage, stdPass));

    ////

    stdScopedEnd;
}

//================================================================
//
// GuiClassImpl::drawNormal
//
//================================================================

stdbool GuiClassImpl::drawNormal(const DrawArgs& args, stdPars(ProcessKit))
{
    stdScopedBegin;

    namespace ms = minimalShell;

    //----------------------------------------------------------------
    //
    // Extend kit.
    //
    //----------------------------------------------------------------

    UserPoint userPoint;
    ms::DesiredOutputSize desiredOutputSize;

    ////

    MsgLogNull msgLogNull;

    auto& oldKit = kit;

    auto kit = kitCombine
    (
        oldKit,
        LocalLogKit{msgLogNull},
        LocalLogAuxKit{false, msgLogNull},
        ms::BaseImageConsolesKit{nullptr, nullptr, nullptr},
        UserPointKit{userPoint},
        ms::DesiredOutputSizeKit{desiredOutputSize}
    );

    //----------------------------------------------------------------
    //
    // Call via minimal shell.
    //
    //----------------------------------------------------------------

    ms::GpuExternalContext externalContext
    {
        kit.gpuProperties,
        kit.gpuCurrentContext,
        kit.gpuCurrentStream
    };

    ////

    guiModule->extendMaxImageSize(args.dstImage.size());

    ////

    GpuAppAllocKit gpuAppAllocKit = kit;

    auto guiModuleThunk = ms::engineModuleThunk
    (
        [&] () {return guiModule->reallocValid();},
        [&] (stdPars(auto)) {return guiModule->realloc(stdPassKit(kitCombine(kit, gpuAppAllocKit)));},
        [&] (stdPars(auto)) {return guiModule->draw(args, stdPass);}
    );

    ////

    bool sysAllocHappened{};

    require(minimalShell->process({&externalContext, guiModuleThunk, guiModuleMemory, true, sysAllocHappened}, stdPass));

    ////

    stdScopedEnd;
}

//================================================================
//
// OverlayBufferHook
//
// Changes the behaviour of moveFrom() and reset().
//
//================================================================

class OverlayBufferHook : public OverlayBuffer
{

public:

    using ImageUser = overlayBuffer::ImageUser;

    OverlayBufferHook(OverlayBuffer& base)
        : base{base} {}

    virtual void clearMemory()
        {return base.clearMemory();}

    ////

    virtual void clearImage()
        {return base.clearImage();}

    virtual stdbool setImage(const Point<Space>& size, const GpuImageProviderBgr32& provider, stdPars(SetImageKit))
        {return base.setImage(size, provider, stdPassThru);}

    virtual stdbool useImage(ImageUser& imageUser, stdPars(UseImageKit))
        {return base.useImage(imageUser, stdPassThru);}

    ////

    virtual bool hasUpdates() const
        {return base.hasUpdates();}

    virtual void reset()
        {}

    virtual bool absorb(OverlayBuffer& other)
        {return base.absorb(other);}

    virtual void moveFrom(OverlayBuffer& other)
        {base.absorb(other);}

private:

    OverlayBuffer& base;

};

//================================================================
//
// GuiClassImpl::processInputUpdates
//
//================================================================

stdbool GuiClassImpl::processInputUpdates(const ProcessInputUpdatesArgs& args, stdPars(ProcessKit))
{

    //----------------------------------------------------------------
    //
    // A hook to avoid the doubling the overlay memory.
    //
    //----------------------------------------------------------------

    auto overlayUpdate = OverlayBufferHook{*overlayBuffer};

    //----------------------------------------------------------------
    //
    // Take all updates.
    //
    //----------------------------------------------------------------

    auto updateBuffers = guiService::BufferRefs
    {
        *shutdownUpdate,
        *globalLogUpdate,
        *localLogUpdate,
        *specialLogUpdate,
        *actionSetUpdate,
        overlayUpdate,
        *configInputUpdate
    };

    args.guiService.takeAllUpdates(updateBuffers);

    //----------------------------------------------------------------
    //
    // * Refresh moments in log updates so that messages display time
    // counts from now instead of message addition time.
    //
    //----------------------------------------------------------------

    auto updateMoment = kit.timer.moment();

    if (updateBuffers.globalLogUpdate.hasUpdates())
        updateBuffers.globalLogUpdate.refreshAllMoments(updateMoment);

    if (updateBuffers.specialLogUpdate.hasUpdates())
        updateBuffers.specialLogUpdate.refreshAllMoments(updateMoment);

    if (args.intrinsicBuffer.hasUpdates())
        args.intrinsicBuffer.refreshAllMoments(updateMoment);

    if (updateBuffers.localLogUpdate.hasUpdates())
        updateBuffers.localLogUpdate.refreshAllMoments(updateMoment);

    //----------------------------------------------------------------
    //
    // Pass global log update to the log keeper.
    //
    //----------------------------------------------------------------

    if (args.intrinsicBuffer.hasUpdates() || globalLogUpdate->hasUpdates())
    {
        logKeeperUpdate->clear();

        ////

        auto makeUpdate = [&] ()
        {
            stdExceptBegin;

            auto receiver = LogBufferReceiver::O | [&] (auto& text, auto& kind, auto& moment)
            {
                logKeeperUpdate->addLine(text.ptr, text.size);
            };

            args.intrinsicBuffer.readLastMessages(receiver, typeMax<size_t>());
            globalLogUpdate->readLastMessages(receiver, typeMax<size_t>());

            stdExceptEnd;
        };

        errorBlock(makeUpdate()); // (update even partially)

        ////

        CHECK(args.logService.addTextUpdate(*logKeeperUpdate));
    }

    //----------------------------------------------------------------
    //
    // Absorb and clear ALL input buffers, except the OVERLAY.
    //
    //----------------------------------------------------------------

    shutdownBuffer->absorb(*shutdownUpdate);

    globalLogBuffer->absorb(args.intrinsicBuffer);
    globalLogBuffer->absorb(*globalLogUpdate);
    globalLogBuffer->absorb(*specialLogUpdate);

    localLogBuffer->absorb(*localLogUpdate);

    bool actionSetIsUpdated = actionSetUpdate->hasUpdates();
    actionSet->absorb(*actionSetUpdate);

    REMEMBER_CLEANUP(configInputUpdate->clearMemory()); // A rare operation, deallocate.

    //----------------------------------------------------------------
    //
    // Action set update handling.
    //
    //----------------------------------------------------------------

    if (actionSetIsUpdated)
    {
        bool ok = true;

        blockExceptBegin
        {
            workerKeyMap.reinitTo(actionSet->dataCount());

            auto receiver = ActionRecordReceiver::O | [&] (ActionId id, CharArray name, CharArray key, CharArray comment)
            {
                blockExceptBegin;

                if (key.size != 0)
                {
                    if_not (workerKeyMap.insert(key, id))
                        printMsg(kit.msgLog, STR("Cannot parse key name \"%0\""), key, msgWarn);
                }

                blockExceptEnd(ok = false);
            };

            actionSet->dataGet(receiver);
        }
        blockExceptEnd(ok = false);

        CHECK_MSG(ok, STR("GUI: Errors in updating of action keymap."));
    }

    //----------------------------------------------------------------
    //
    // Config update handling.
    //
    //----------------------------------------------------------------

    if (configInputUpdate->hasUpdates())
    {
        using namespace cfgSerializeImpl;
        errorBlock(loadVarsFromTree({args.guiSerialization, *configInputUpdate, configTemp, false, true}, stdPassNc));

        CHECK(args.workerService.addConfigUpdate(*configInputUpdate)); // Send to WORKER
    }

    returnTrue;
}

//================================================================
//
// GuiClassImpl::updateAndRedraw
//
//================================================================

stdbool GuiClassImpl::updateAndRedraw(const UpdateAndRedrawArgs& args, stdPars(ProcessKit))
{
    stdScopedBegin;

    //----------------------------------------------------------------
    //
    // Update to worker.
    //
    //----------------------------------------------------------------

    if_not (desiredOutputSize && allv(*desiredOutputSize == args.dstImage.size()))
    {
        updatesToWorker.displaySettingsUpdate.set({args.dstImage.size()});
        args.workerService.addDisplaySettingsUpdate(updatesToWorker.displaySettingsUpdate);
        desiredOutputSize = args.dstImage.size();
    }

    //----------------------------------------------------------------
    //
    // Update from worker.
    //
    //----------------------------------------------------------------

    errorBlock(processInputUpdates({args.guiService, args.workerService, args.logService, args.intrinsicBuffer, args.guiSerialization}, stdPassNc));

    //----------------------------------------------------------------
    //
    // Draw. Recovery point: If normal drawing fails, draw error pattern.
    //
    //----------------------------------------------------------------

    auto drawArgs = DrawArgs{*overlayBuffer, *globalLogBuffer, *localLogBuffer, args.dstImage};

    if_not (errorBlock(drawNormal(drawArgs, stdPassNc)))
        require(drawErrorPattern(drawArgs, stdPass));

    ////

    stdScopedEnd;
}

//================================================================
//
// GuiClassImpl::printHelp
//
//================================================================

stdbool GuiClassImpl::printHelp(const ProcessArgs& args, bool printAlgoKeys, stdPars(ProcessKit))
{
    stdExceptBegin;

    //
    // Header.
    //

    REMEMBER_CLEANUP(printMsg(kit.msgLog, STR("")));

    if_not (printAlgoKeys)
    {
        printMsg(kit.msgLog, STR("Graphical shell documentation can be found at the root: /testbedGL/README.md."));
        printMsg(kit.msgLog, STR("You can read it directly on the GitLab website."));
        printMsg(kit.msgLog, STR(""));
    }

    //
    // Key printer.
    //

    auto receiver = signalTools::ActionReceiver::O | [&] (ActionId id, CharArray name, CharArray key, CharArray comment)
    {
        ensurev(key.size != 0);
        printMsg(kit.msgLog, STR("%1: %0"), key, comment.size ? comment : name);
    };

    //
    // Shell keys.
    //

    if_not (printAlgoKeys)
    {
        printMsg(kit.msgLog, STR("Graphical shell keys:"));
        printMsg(kit.msgLog, STR(""));

        size_t signalCount{};

        require(signalTools::gatherActionSet(args.guiSerialization, receiver, signalCount, stdPass));

        printMsg(kit.msgLog, STR(""));

        printMsg(kit.msgLog, STR("Press Alt+F1 to view the algorithm module keys (the list might be long)."));
    }

    //
    // Algorithm keys.
    //

    if (printAlgoKeys)
    {
        printMsg(kit.msgLog, STR("Algorithm module keys:"));
        printMsg(kit.msgLog, STR(""));

        actionSet->dataGet(receiver);
    }

    ////

    stdExceptEnd;
}

//================================================================
//
// GuiClassImpl::processEvents
//
//================================================================

stdbool GuiClassImpl::processEvents(const ProcessArgs& args, stdPars(ProcessKit))
{
    REQUIRE(initialized);

    //----------------------------------------------------------------
    //
    // Update config of the GUI thread intrinsic delta-buffer of the global log.
    //
    //----------------------------------------------------------------

    args.intrinsicBufferDebuggerOutputControl.setDebuggerOutputLevel(gLogDebuggerOutputLevel);

    //----------------------------------------------------------------
    //
    // Before waiting user events, send the updated config variables
    // of the GuiClass and the external shell to ConfigKeeper.
    //
    //----------------------------------------------------------------

    auto updateConfigVars = [&] ()
    {
        using namespace cfgSerializeImpl;

        require(saveVarsToTree({args.guiSerialization, *configOutputBuffer, configTemp, true, true, false}, stdPass));

        ////

        if (configOutputBuffer->hasUpdates())
            REQUIRE(args.configService.addConfigUpdate(*configOutputBuffer));

        returnTrue;
    };

    errorBlock(updateConfigVars());

    //----------------------------------------------------------------
    //
    // Event receivers.
    //
    //----------------------------------------------------------------

    auto keyReceiver = KeyReceiver::O | [&] (const KeyEvent& event, stdParsNull)
    {
        require(keyBuffer->receiveKey(event, stdPass));

        returnTrue;
    };

    ////

    auto refreshReceiver = RefreshReceiver::O | [&] (stdParsNull)
    {
        auto drawer = Drawer::O | [&] (auto& dstImage, stdParsNull)
        {
            return updateAndRedraw
            (
                {args.guiService, args.workerService, args.logService, args.intrinsicBuffer, args.guiSerialization, dstImage},
                stdPass
            );
        };

        errorBlock(args.drawReceiver(drawer, stdPassNullNc));

        returnTrue;
    };

    ////

    auto mouseMoveReceiver = MouseMoveReceiver::O | [&] (auto& pos, stdParsNull)
    {
        RedrawRequest redraw;

        errorBlock(guiModule->mouseMoveReceiver(pos, redraw, stdPassNc));

        if (redraw.on)
            require(refreshReceiver(stdPass));

        auto offset = guiModule->getOverlayOffset();

        if (offset)
            mousePointer.position = *offset + pos;

        returnTrue;
    };

    ////

    auto mouseButtonReceiver = MouseButtonReceiver::O | [&] (const MouseButtonEvent& event, stdParsNull)
    {
        RedrawRequest redraw;

        errorBlock(guiModule->mouseButtonReceiver(event, redraw, stdPassNc));

        if (redraw.on)
            require(refreshReceiver(stdPass));

        if (event.button == 0)
            mousePointer.button0 = event.press;

        if (event.button == 1)
            mousePointer.button1 = event.press;

        returnTrue;
    };

    ////

    auto scrollReceiver = ScrollReceiver::O | [&] (auto& event, stdParsNull)
    {
        returnTrue;
    };

    ////

    auto resizeReceiver = ResizeReceiver::O | [&] (auto& event, stdParsNull)
    {
        returnTrue;
    };

    ////

    auto receivers = kitCombine
    (
        RefreshReceiverKit{refreshReceiver},
        KeyReceiverKit{keyReceiver},
        MouseMoveReceiverKit{mouseMoveReceiver},
        MouseButtonReceiverKit{mouseButtonReceiver},
        ScrollReceiverKit{scrollReceiver},
        ResizeReceiverKit{resizeReceiver}
    );

    //----------------------------------------------------------------
    //
    // Decide on the waiting mode.
    //
    //----------------------------------------------------------------

    // Wait timeout. If not empty, it means that animation is happening.
    OptionalObject<uint32> waitTimeoutMs;

    ////

    require(guiModule->checkWake({*globalLogBuffer}, stdPass));

    auto wakeMoment = guiModule->getWakeMoment();

    if (wakeMoment)
    {
        auto remainingTime = kit.timer.diff(kit.timer.moment(), *wakeMoment);
        REQUIRE(def(remainingTime));
        remainingTime = clampMin(remainingTime, 0.f);
        uint32 timeoutMs = 0;
        REQUIRE(convertUp(remainingTime * 1000.f, timeoutMs));
        waitTimeoutMs = timeoutMs;
    }

    ////

    if (args.intrinsicBuffer.hasUpdates())
        waitTimeoutMs = 0;

    ////

    bool noWait = (waitTimeoutMs && *waitTimeoutMs == 0);

    require(args.eventSource(!noWait, waitTimeoutMs, receivers, stdPass));

    //----------------------------------------------------------------
    //
    // Keys to WORKER actions.
    //
    //----------------------------------------------------------------

    auto keys = keyBuffer->getBuffer();

    if (keys.size())
    {
        bool perfect = true;

        blockExceptBegin
        {
            ARRAY_EXPOSE(keys);

            for_count (i, keysSize)
            {
                const KeyEvent& key = keysPtr[i];

                if_not (key.action != KeyAction::Release && key.code != 0)
                    continue;

                ActionId id{};

                if (workerKeyMap.find(key, id))
                    perfect &= updatesToWorker.actionsUpdate.dataAdd(id);
            }
        }
        blockExceptEnd(perfect = false);

        CHECK_MSG(perfect, STR("GUI: Errors in passing actions to WORKER."));
    }

    //----------------------------------------------------------------
    //
    // Mouse events to WORKER?
    //
    //----------------------------------------------------------------

    bool workerMouse = false;

    if (workerByMouse == WorkerByMouse::All)
        workerMouse = mousePointer.hasUpdates();

    if (workerByMouse == WorkerByMouse::PressAndRelease)
        workerMouse = mousePointer.button0 || mousePointer.button1;

    if (workerByMouse == WorkerByMouse::Press)
    {
        workerMouse =
            (mousePointer.button0 && *mousePointer.button0) ||
            (mousePointer.button1 && *mousePointer.button1);
    }

    //----------------------------------------------------------------
    //
    // Feed updates to WORKER.
    //
    //----------------------------------------------------------------

    if (updatesToWorker.hasUpdates() || mousePointer.hasUpdates())
    {
        bool notify = updatesToWorker.hasUpdates() || workerMouse;

        ////

        updatesToWorker.mousePointerUpdate.set(mousePointer);
        mousePointer = {};

        ////

        bool ok = args.workerService.addAllUpdates(updatesToWorker, notify);
        CHECK_MSG(ok, STR("GUI: Errors in passing input update to WORKER"));
    }

    //----------------------------------------------------------------
    //
    // Redraw?
    //
    //----------------------------------------------------------------

    bool needRedraw = false;

    if (waitTimeoutMs)
        needRedraw = true;

    if (args.guiService.checkUpdates())
        needRedraw = true;

    //----------------------------------------------------------------
    //
    // Feed received keys to signals.
    //
    //----------------------------------------------------------------

    signalSupport.feedSignals(keyBuffer->getBuffer(), args.guiSerialization);

    keyBuffer->clearBuffer();

    //----------------------------------------------------------------
    //
    // Process signals.
    //
    //----------------------------------------------------------------

    if (exitSignal || shutdownBuffer->hasUpdates())
    {
        args.shutdownRequest.on = true;
    }

    ////

    auto makeEditRequest = [&] (stdPars(auto))
    {
        if (textEditor().size() == 0)
        {
            printMsg(kit.msgLog, textEditorHelpMessage(), msgWarn);
            returnFalse;
        }

        editRequest->addRequest(textEditor(), textEditorHelpMessage());
        returnTrue;
    };

    ////

    if (editConfig)
    {
        errorBlock(makeEditRequest(stdPassNc)) &&
        args.configService.addEditRequest(*editRequest);
    }

    ////

    if (editLog)
    {
        errorBlock(makeEditRequest(stdPassNc)) &&
        args.logService.addEditRequest(*editRequest);
    }

    ////

    if (welcomeMessageToggle)
    {
        welcomeMessage = (welcomeMessage() ^ (welcomeMessageToggle % 2)) != 0;
        printMsg(kit.msgLog, STR("Welcome message %."), welcomeMessage ? STR("enabled") : STR("disabled"));
    }

    ////

    if (printHelpSignal || printAlgoKeysSignal)
    {
        auto body = [&] ()
        {
            require(printHelp(args, printAlgoKeysSignal != 0, stdPass));

            require(refreshReceiver(stdPass));

            require(makeEditRequest(stdPass));
            REQUIRE(args.logService.addEditRequest(*editRequest));

            returnTrue;
        };

        errorBlock(body());
    }

    //----------------------------------------------------------------
    //
    // Redraw
    //
    //----------------------------------------------------------------

    if (needRedraw)
        require(refreshReceiver(stdPass));

    ////

    returnTrue;
}

//----------------------------------------------------------------

}
