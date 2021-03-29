#pragma once

#include "allocation/mallocKit.h"
#include "configFile/cfgSerialization.h"
#include "interfaces/threadManagerKit.h"
#include "kits/moduleHeader.h"
#include "memController/memController.h"
#include "storage/smartPtr.h"
#include "interfaces/fileToolsKit.h"
#include "baseInterfaces/baseImageConsole.h"

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

    virtual void setImageSavingActive(bool active) =0;
    virtual void setImageSavingDir(const CharType* dir) =0; // can be NULL

    virtual void setImageSavingLockstepCounter(uint32 counter) =0;
    virtual const CharType* getImageSavingDir() const =0;

public:

    virtual void serialize(const CfgSerializeKit& kit) =0;

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
