#pragma once

#include "gpuModuleHeader.h"

namespace halfFloatTest {

//================================================================
//
// Process
// ProcessKit
//
//================================================================

struct Process {};
using ProcessKit = KitCombine<ModuleProcessKit, GpuAppExecKit>;
using ReallocKit = KitCombine<ModuleReallocKit, GpuAppExecKit>;

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
