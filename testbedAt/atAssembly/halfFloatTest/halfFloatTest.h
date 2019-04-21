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
    bool reallocValid() const;
    stdbool realloc(stdPars(ReallocKit));
    stdbool process(const Process& o, stdPars(ProcessKit));

private:

    StaticClass<class HalfFloatTestImpl, 256> instance;

};

//----------------------------------------------------------------

}
