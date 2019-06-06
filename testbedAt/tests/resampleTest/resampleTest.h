#pragma once

#include "gpuModuleHeader.h"
#include "storage/dynamicClass.h"
#include "kits/gpuRgbFrameKit.h"

namespace resampleTest {

//================================================================
//
// ProcessParams
//
//================================================================

using ProcessParams = GpuRgbFrameKit;

//================================================================
//
// ResampleTest
//
//================================================================

class ResampleTest
{

public:

    ResampleTest();
    ~ResampleTest();

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const;

public:

    stdbool process(const ProcessParams& o, stdPars(GpuModuleProcessKit));

private:

    DynamicClass<class ResampleTestImpl> instance;

};

//----------------------------------------------------------------

}
