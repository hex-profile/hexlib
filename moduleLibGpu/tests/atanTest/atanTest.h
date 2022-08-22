#pragma once

#include "gpuModuleHeader.h"
#include "storage/smartPtr.h"

namespace atanTest {

//================================================================
//
// AtanTest
//
//================================================================

struct AtanTest
{
    static UniquePtr<AtanTest> create();
    virtual ~AtanTest() {}

    virtual void serialize(const ModuleSerializeKit& kit) =0;
    virtual bool active() const =0;
    virtual stdbool process(stdPars(GpuModuleProcessKit)) =0;
};

//----------------------------------------------------------------

}

using atanTest::AtanTest;
