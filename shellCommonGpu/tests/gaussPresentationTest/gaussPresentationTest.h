#pragma once

#include "gpuModuleHeader.h"

namespace gaussPresentationTest {

//================================================================
//
// ProcessParams
//
//================================================================

struct ProcessParams {};

//================================================================
//
// GaussPresentationTest
//
//================================================================

class GaussPresentationTest
{

public:

    GaussPresentationTest();
    ~GaussPresentationTest();

    void serialize(const ModuleSerializeKit& kit);
    bool active() const;
    void realloc(stdPars(GpuModuleReallocKit));
    bool reallocValid() const;
    void process(const ProcessParams& o, stdPars(GpuModuleProcessKit));

private:

    DynamicClass<class GaussPresentationTestImpl> instance;

};

//----------------------------------------------------------------

}
