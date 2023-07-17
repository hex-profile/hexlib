#pragma once

#include "allocation/mallocKit.h"
#include "baseInterfaces/baseImageConsole.h"
#include "kits/moduleHeader.h"
#include "memController/memController.h"
#include "minimalShell/minimalShellTypes.h"
#include "storage/smartPtr.h"
#include "userOutput/diagnosticKit.h"
#include "imageConsole/gpuImageConsole.h"

namespace minimalShell {

//================================================================
//
// BaseImageConsolesKit
//
// Pointers can be NULL.
//
//================================================================

KIT_CREATE3
(
    BaseImageConsolesKit,
    GpuBaseConsole*, gpuBaseConsole,
    BaseImageConsole*, baseImageConsole,
    BaseVideoOverlay*, baseVideoOverlay
);

//================================================================
//
// MinimalShell
//
//================================================================

struct MinimalShell
{

    //----------------------------------------------------------------
    //
    // Creation.
    //
    //----------------------------------------------------------------

    static UniquePtr<MinimalShell> create();

    virtual ~MinimalShell() {}

    //----------------------------------------------------------------
    //
    // Settings.
    //
    //----------------------------------------------------------------

    virtual void serialize(const CfgSerializeKit& kit) =0;

    virtual Settings& settings() =0;

    //----------------------------------------------------------------
    //
    // Init / deinit state (apart from settings).
    //
    // The initialization may be done in two modes:
    // * The shell is the maintainer of gpu context and stream OR
    // * The shell will use externally provided gpu context and stream.
    //
    // The initialization is controlled by "gpu context maintainer" setting.
    //
    // If the shell is initialized in the external mode, the user has to pass
    // non-null external context pointer to processing functions.
    //
    //----------------------------------------------------------------

    using InitKit = KitCombine<DiagnosticKit, MallocKit>;

    virtual stdbool init(stdPars(InitKit)) =0;

    //----------------------------------------------------------------
    //
    // Process.
    //
    //----------------------------------------------------------------

    struct ProcessArgs
    {
        const GpuExternalContext* externalContext;
        EngineModule& engineModule;
        MemController& engineMemory;
        bool runExecutionPhase;
        bool& sysAllocHappened;
    };

    using BaseProcessKit = KitCombine<ErrorLogKit, MsgLogsKit, MsgLogExKit, TimerKit, MallocKit>;

    using ProcessKit = KitCombine<BaseProcessKit, BaseImageConsolesKit, UserPointKit, DesiredOutputSizeKit>;

    virtual stdbool process(const ProcessArgs& args, stdPars(ProcessKit)) =0;

    //----------------------------------------------------------------
    //
    // Profiling.
    //
    //----------------------------------------------------------------

    virtual bool profilingActive() const =0;

    using ReportKit = BaseProcessKit;

    virtual stdbool profilingReport(const GpuExternalContext* externalContext, stdPars(ReportKit)) =0;

};

//----------------------------------------------------------------

}
