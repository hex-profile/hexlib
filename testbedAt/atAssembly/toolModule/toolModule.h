#pragma once

#include "atAssembly/videoPreprocessor/videoPreprocessor.h"

namespace toolModule {

//================================================================
//
// ToolTarget
//
//================================================================

using ToolTarget = videoPreprocessor::VideoPrepTarget;
using ToolTargetProcessKit = videoPreprocessor::ProcessTargetKit;

//================================================================
//
// ReallocKit
// ProcessKit
//
//================================================================

using ReallocKit = videoPreprocessor::ReallocKit;
using ProcessKit = videoPreprocessor::ProcessKit;

//================================================================
//
// ToolModule
//
//================================================================

class ToolModule
{

public:

    ToolModule();
    ~ToolModule();

    void serialize(const ModuleSerializeKit& kit);
    void setFrameSize(const Point<Space>& frameSize);

    bool reallocValid() const;
    stdbool realloc(stdPars(ReallocKit));

    Point<Space> outputFrameSize() const;
    stdbool process(ToolTarget& target, stdPars(ProcessKit));

private:

    DynamicClass<class ToolModuleImpl> instance;

};

//----------------------------------------------------------------

}
