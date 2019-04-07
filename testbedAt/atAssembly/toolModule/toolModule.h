#pragma once

#include "atAssembly/toolModule/gpuOverheadTest.h"
#include "atAssembly/videoPreprocessor/videoPreprocessor.h"

//================================================================
//
// ToolTarget
//
//================================================================

using ToolTarget = videoPreprocessor::VideoPrepTarget;
using ToolTargetProcessKit = videoPreprocessor::ProcessTargetKit;

//================================================================
//
// ToolModule
//
//================================================================

class ToolModule
{

public:

    KIT_COMBINE3(ReallocKit, ModuleReallocKit, GpuAppExecKit, AtCommonKit);
    KIT_COMBINE7(ProcessKit, ModuleProcessKit, GpuAppExecKit, FileToolsKit, MallocKit, AtProcessKit, FrameAdvanceKit, ThreadManagerKit);

    void serialize(const ModuleSerializeKit& kit);

public:

    void setFrameSize(const Point<Space>& frameSize)
        {this->frameSize = frameSize;}

    Point<Space> outputFrameSize() const
        {return videoPreprocessor.outputFrameSize();}

public:

    bool reallocValid() const;
    bool realloc(stdPars(ReallocKit));
    bool process(ToolTarget& toolTarget, stdPars(ProcessKit));

private:

    Point<Space> allocFrameSize = point(0);
    Point<Space> frameSize = point(0);

    videoPreprocessor::VideoPreprocessor videoPreprocessor;
    GpuOverheadTest gpuOverheadTest;

};
