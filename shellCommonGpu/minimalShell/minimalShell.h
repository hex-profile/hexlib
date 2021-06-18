#pragma once

#include "allocation/mallocKit.h"
#include "baseInterfaces/baseImageConsole.h"
#include "configFile/cfgSerialization.h"
#include "interfaces/fileToolsKit.h"
#include "interfaces/threadManagerKit.h"
#include "kits/moduleHeader.h"
#include "memController/memController.h"
#include "storage/smartPtr.h"
#include "minimalShell/minimalShellTypes.h"

namespace minimalShell {

//================================================================
//
// ParamsKit
//
//================================================================

KIT_CREATE4(ParamsKit, EngineModule&, engineModule, MemController&, engineMemory, bool, runExecutionPhase, bool&, sysAllocHappened);

//================================================================
//
// BaseImageConsolesKit
//
// Pointers can be NULL.
//
//================================================================

KIT_CREATE2(BaseImageConsolesKit, BaseImageConsole*, baseImageConsole, BaseVideoOverlay*, baseVideoOverlay);

//================================================================
//
// MinimalShell
//
//================================================================

class MinimalShell : public CfgSerialization
{

public:

    static UniquePtr<MinimalShell> create();
    virtual ~MinimalShell() {}

public:

    virtual void serialize(const CfgSerializeKit& kit) =0;

    virtual Settings& settings() =0;

public:

    virtual bool isInitialized() const =0;

    using InitKit = KitCombine<ErrorLogKit, MsgLogsKit, ErrorLogExKit, TimerKit, MallocKit, ThreadManagerKit, FileToolsKit>;
    virtual stdbool init(stdPars(InitKit)) =0;

public:

    using ProcessKit = KitCombine<InitKit, BaseImageConsolesKit, UserPointKit>;

    stdbool process(EngineModule& engineModule, MemController& engineMemory, bool runExecutionPhase, bool& sysAllocHappened, stdPars(ProcessKit))
        {return processEntry(stdPassKit(kitCombine(kit, ParamsKit(engineModule, engineMemory, runExecutionPhase, sysAllocHappened))));}

public:

    using ProcessEntryKit = KitCombine<ProcessKit, ParamsKit>;

    virtual stdbool processEntry(stdPars(ProcessEntryKit)) =0;

public:

    virtual bool profilingActive() const =0;

    using ReportKit = InitKit;

    virtual stdbool profilingReport(stdPars(ReportKit)) =0;

};

//----------------------------------------------------------------

}
