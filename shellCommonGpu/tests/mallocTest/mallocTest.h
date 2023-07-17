#pragma once

#include "gpuModuleHeader.h"
#include "storage/smartPtr.h"

namespace mallocTest {

//================================================================
//
// MallocTest
//
//================================================================

struct MallocTest
{
    static UniquePtr<MallocTest> create();
    virtual ~MallocTest() {}

    virtual void serialize(const ModuleSerializeKit& kit) =0;
    virtual bool active() const =0;
    virtual stdbool process(stdPars(GpuModuleProcessKit)) =0;
};

//----------------------------------------------------------------

}

using mallocTest::MallocTest;
