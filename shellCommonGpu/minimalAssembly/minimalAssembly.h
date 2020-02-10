#pragma once

#include "allocation/mallocKit.h"
#include "configFile/cfgSerialization.h"
#include "interfaces/threadManagerKit.h"
#include "kits/moduleHeader.h"
#include "memController/memController.h"

namespace minimalAssembly {

//================================================================
//
// InitKit
// ProcessKit
//
//================================================================

using InitKit = KitCombine<ErrorLogKit, MsgLogsKit, ErrorLogExKit, TimerKit, MallocKit, ThreadManagerKit>;
using ProcessKit = InitKit;

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
// MinimalAssembly
//
//================================================================

class MinimalAssembly : public CfgSerialization
{

public:

    MinimalAssembly();
    ~MinimalAssembly();

public:

    void serialize(const CfgSerializeKit& kit);

public:

    stdbool init(stdPars(InitKit));

public:

    stdbool process(EngineModule& engineModule, MemController& engineMemory, stdPars(ProcessKit));

private:

    DynamicClass<class MinimalAssemblyImpl> instance;

};

//----------------------------------------------------------------

}
