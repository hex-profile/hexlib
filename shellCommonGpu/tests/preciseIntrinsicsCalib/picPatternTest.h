#pragma once

#include "gpuModuleHeader.h"
#include "storage/smartPtr.h"

namespace picPatternTest {

//================================================================
//
// PicPatternTest
//
//================================================================

struct PicPatternTest
{
    static UniquePtr<PicPatternTest> create();
    virtual ~PicPatternTest() {}

    ////

    virtual void serialize(const ModuleSerializeKit& kit) =0;

    virtual bool active() const =0;

    ////

    virtual void process(stdPars(GpuModuleProcessKit)) =0;
};

//----------------------------------------------------------------

}

using picPatternTest::PicPatternTest;
