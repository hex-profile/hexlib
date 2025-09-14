#pragma once

#include "gpuModuleHeader.h"
#include "userOutput/diagnosticKit.h"

namespace floatRangesTest {

//================================================================
//
// Process
// ProcessKit
//
//================================================================

using ProcessKit = DiagnosticKit;

//================================================================
//
// FloatRangesTest
//
//================================================================

class FloatRangesTest
{

public:

    FloatRangesTest();
    ~FloatRangesTest();

public:

    void serialize(const ModuleSerializeKit& kit);
    void process(stdPars(ProcessKit));

private:

    DynamicClass<class FloatRangesTestImpl> instance;

};

//----------------------------------------------------------------

}
