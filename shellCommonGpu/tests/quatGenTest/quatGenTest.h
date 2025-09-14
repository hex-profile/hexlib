#pragma once

#include "gpuModuleHeader.h"
#include "storage/smartPtr.h"

namespace quatGenTest {

//================================================================
//
// QuatGenTest
//
//================================================================

struct QuatGenTest
{
    static UniquePtr<QuatGenTest> create();
    virtual ~QuatGenTest() {}

    virtual void serialize(const ModuleSerializeKit& kit) =0;
    virtual bool active() const =0;
    virtual stdbool process(stdPars(GpuModuleProcessKit)) =0;
};

//----------------------------------------------------------------

}

using quatGenTest::QuatGenTest;
