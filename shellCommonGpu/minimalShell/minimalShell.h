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
// EngineKit
//
//================================================================

KIT_CREATE2(EngineKit, EngineModule&, engineModule, MemController&, engineMemory);

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

    stdbool process(EngineModule& engineModule, MemController& engineMemory, stdPars(ProcessKit))
        {return processEntry(stdPassKit(kitCombine(kit, EngineKit(engineModule, engineMemory))));}

public:

    using ProcessEntryKit = KitCombine<ProcessKit, EngineKit>;

    virtual stdbool processEntry(stdPars(ProcessEntryKit)) =0;

};

//----------------------------------------------------------------

}
