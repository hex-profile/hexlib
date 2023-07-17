#pragma once

#include "gpuModuleHeader.h"
#include "storage/smartPtr.h"

namespace formatTest {

//================================================================
//
// FormatTest
//
//================================================================

struct FormatTest
{
    static UniquePtr<FormatTest> create();
    virtual ~FormatTest() {}

    virtual void serialize(const ModuleSerializeKit& kit) =0;
    virtual bool active() const =0;
    virtual stdbool process(stdPars(GpuModuleProcessKit)) =0;
};

//----------------------------------------------------------------

}

using formatTest::FormatTest;
