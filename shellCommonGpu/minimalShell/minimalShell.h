#pragma once

#include "allocation/mallocKit.h"
#include "configFile/cfgSerialization.h"
#include "interfaces/threadManagerKit.h"
#include "kits/moduleHeader.h"
#include "memController/memController.h"
#include "storage/smartPtr.h"
#include "interfaces/fileTools.h"

namespace minimalShell {

//================================================================
//
// EngineReallocKit
// EngineProcessKit
//
//================================================================

using EngineReallocKit = KitCombine<GpuModuleReallocKit, GpuBlockAllocatorKit>;
using EngineProcessKit = KitCombine<GpuModuleProcessKit, GpuBlockAllocatorKit>;

//================================================================
//
// EngineModule
//
//================================================================

struct EngineModule
{
    virtual bool reallocValid() const =0;
    virtual stdbool realloc(stdPars(EngineReallocKit)) =0;

    virtual stdbool process(stdPars(EngineProcessKit)) =0;
};

//================================================================
//
// ParamsKit
//
//================================================================

KIT_CREATE3(ParamsKit, EngineModule&, engineModule, MemController&, engineMemory, bool, runExecutionPhase);

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

public:

    using InitKit = KitCombine<ErrorLogKit, MsgLogsKit, ErrorLogExKit, TimerKit, MallocKit, ThreadManagerKit, FileToolsKit>;
    virtual stdbool init(stdPars(InitKit)) =0;

public:

    using ProcessKit = InitKit;

    stdbool process(EngineModule& engineModule, MemController& engineMemory, bool runExecutionPhase, stdPars(ProcessKit))
        {return processEntry(stdPassKit(kitCombine(kit, ParamsKit(engineModule, engineMemory, runExecutionPhase))));}

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
