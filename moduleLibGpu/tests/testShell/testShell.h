#pragma once

#include "gpuModuleHeader.h"
#include "storage/smartPtr.h"
#include "kits/gpuRgbFrameKit.h"

namespace testShell {

//================================================================
//
// Process
//
//================================================================

struct Process
{
    virtual stdbool process(stdNullPars) const =0;
};

//================================================================
//
// processByLambda
//
//================================================================

template <typename Lambda>
class ProcessByLambda : public Process
{

public:

    ProcessByLambda(const Lambda& lambda)
        : lambda{lambda} {}

    virtual stdbool process(stdNullPars) const
        {return lambda(stdNullPass);}

private:

    Lambda lambda;

};

//----------------------------------------------------------------

template <typename Lambda>
inline auto processByLambda(const Lambda& lambda)
    {return ProcessByLambda<Lambda>{lambda};}

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
    virtual stdbool process(const Process& base, stdPars(ProcessKit)) =0;
};

//----------------------------------------------------------------

}

using testShell::TestShell;

