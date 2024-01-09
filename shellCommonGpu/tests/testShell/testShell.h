#pragma once

#include "gpuModuleHeader.h"
#include "storage/smartPtr.h"
#include "kits/gpuRgbFrameKit.h"
#include "storage/adapters/callable.h"

namespace testShell {

//================================================================
//
// Process
//
//================================================================

using Process = Callable<stdbool (stdParsNull)>;

//================================================================
//
// TestShell
//
//================================================================

struct TestShell
{
    static UniquePtr<TestShell> create();
    virtual ~TestShell() {}

    virtual void serialize(const ModuleSerializeKit& kit) =0;

    using ProcessKit = KitCombine<GpuModuleProcessKit, GpuRgbFrameKit>;
    virtual stdbool process(Process& baseProcess, stdPars(ProcessKit)) =0;
};

//----------------------------------------------------------------

}

using testShell::TestShell;

