#pragma once

#include "allocation/mallocKit.h"
#include "channels/buffers/logBuffer/logBuffer.h"
#include "channels/configService/configService.h"
#include "channels/guiService/guiService.h"
#include "channels/logService/logService.h"
#include "channels/workerService/workerService.h"
#include "cfg/cfgSerialization.h"
#include "gpuLayer/gpuLayerKits.h"
#include "lib/eventReceivers.h"
#include "lib/logToBuffer/debuggerOutputControl.h"
#include "minimalShell/minimalShellTypes.h"
#include "stdFunc/stdFunc.h"
#include "storage/adapters/lambdaThunk.h"
#include "storage/smartPtr.h"
#include "userOutput/diagnosticKit.h"

namespace gui {

//================================================================
//
// InitArgs
//
//================================================================

struct InitArgs
{
    Point<Space> const maxExpectedOutputResolution;

    CfgSerialization& guiSerialization; // GuiClass + the external shell
};

//================================================================
//
// EventSource
//
//================================================================

using EventSource = Callable<stdbool (bool waitEvents, const OptionalObject<uint32>& waitTimeoutMs, const EventReceivers& receivers, stdParsNull)>;

//================================================================
//
// Drawer
//
//================================================================

using Drawer = Callable<stdbool (const GpuMatrix<uint8_x4>& dstImage, stdParsNull)>;

//================================================================
//
// DrawReceiver
//
//================================================================

using DrawReceiver = Callable<stdbool (Drawer& drawer, stdParsNull)>;

//================================================================
//
// ShutdownRequest
//
//================================================================

struct ShutdownRequest
{
    bool on = false;
};

//================================================================
//
// GuiClass
//
//================================================================

struct GuiClass
{
    static UniquePtr<GuiClass> create();
    virtual ~GuiClass() {}

    //----------------------------------------------------------------
    //
    // Config.
    //
    //----------------------------------------------------------------

    virtual void serialize(const CfgSerializeKit& kit) =0;

    //----------------------------------------------------------------
    //
    // Init.
    //
    //----------------------------------------------------------------

    using InitKit = KitCombine<DiagnosticKit, MallocKit, TimerKit>;

    virtual stdbool init(const InitArgs& args, stdPars(InitKit)) =0;

    ////

    virtual void takeGlobalLog(LogBuffer& result) =0;

    //----------------------------------------------------------------
    //
    // Process.
    //
    //----------------------------------------------------------------

    using GpuBasicKit = KitCombine<GpuInitKit, GpuPropertiesKit, GpuCurrentContextKit, GpuExecKit, GpuCurrentStreamKit>;

    using ProcessKit = KitCombine<DiagnosticKit, MallocKit, TimerKit, GpuBasicKit>;

    struct ProcessArgs
    {
        LogBuffer& intrinsicBuffer;
        DebuggerOutputControl& intrinsicBufferDebuggerOutputControl;

        EventSource& eventSource;
        DrawReceiver& drawReceiver;
        ShutdownRequest& shutdownRequest;

        guiService::ServerApi& guiService;
        workerService::ClientApi& workerService;
        configService::ClientApi& configService;
        logService::ClientApi& logService;

        CfgSerialization& guiSerialization; // GuiClass + the external shell
    };

    virtual stdbool processEvents(const ProcessArgs& args, stdPars(ProcessKit)) =0;

};

//----------------------------------------------------------------

}

using gui::GuiClass;
