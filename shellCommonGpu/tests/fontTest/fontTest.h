#pragma once

#include "gpuModuleHeader.h"
#include "storage/smartPtr.h"

namespace fontTest {

//================================================================
//
// FontTest
//
//================================================================

struct FontTest
{
    static UniquePtr<FontTest> create();
    virtual ~FontTest() {}

    virtual void serialize(const ModuleSerializeKit& kit) =0;
    virtual bool active() const =0;
    virtual stdbool process(stdPars(GpuModuleProcessKit)) =0;
};

//----------------------------------------------------------------

}

using fontTest::FontTest;
