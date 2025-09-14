#pragma once

#include "gpuModuleHeader.h"
#include "storage/dynamicClass.h"
#include "kits/gpuRgbFrameKit.h"

namespace resamplingTest {

//================================================================
//
// ProcessParams
//
//================================================================

using ProcessParams = GpuRgbFrameKit;

//================================================================
//
// ResamplingTest
//
//================================================================

class ResamplingTest
{

public:

    ResamplingTest();
    ~ResamplingTest();

public:

    void serialize(const ModuleSerializeKit& kit);
    bool active() const;

public:

    void process(const ProcessParams& o, stdPars(GpuModuleProcessKit));

private:

    DynamicClass<class ResamplingTestImpl> instance;

};

//----------------------------------------------------------------

}
