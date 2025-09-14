#pragma once

#include "gpuModuleHeader.h"
#include "storage/smartPtr.h"

namespace popCountTest {

//================================================================
//
// PopCountTest
//
//================================================================

struct PopCountTest
{
    static UniquePtr<PopCountTest> create();
    virtual ~PopCountTest() {}

    virtual void serialize(const ModuleSerializeKit& kit) =0;
    virtual bool active() const =0;
    virtual void process(stdPars(GpuModuleProcessKit)) =0;
};

//----------------------------------------------------------------

}

using popCountTest::PopCountTest;
