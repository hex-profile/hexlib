#pragma once

#include "gpuModuleHeader.h"

namespace halfFloatTest {

//================================================================
//
// Process
// ProcessKit
//
//================================================================

KIT_CREATE0(Process);
KIT_COMBINE2(ProcessKit, ModuleProcessKit, GpuAppExecKit);
KIT_COMBINE2(ReallocKit, ModuleReallocKit, GpuAppExecKit);

//================================================================
//
// HalfFloatTest
//
//================================================================

class HalfFloatTest
{

public:

    HalfFloatTest();
    ~HalfFloatTest();

public:

    void serialize(const ModuleSerializeKit& kit);
    stdbool process(const Process& o, stdPars(ProcessKit));

private:

    DynamicClass<class HalfFloatTestImpl> instance;

};

//----------------------------------------------------------------

}
