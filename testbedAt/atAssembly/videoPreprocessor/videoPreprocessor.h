#pragma once

#include "allocation/mallocKit.h"
#include "atAssembly/frameAdvanceKit.h"
#include "atInterface/atInterfaceKit.h"
#include "gpuModuleHeader.h"
#include "interfaces/fileToolsKit.h"
#include "kits/alternativeVersionKit.h"
#include "kits/displayParamsKit.h"
#include "kits/gpuRgbFrameKit.h"
#include "kits/inputVideoNameKit.h"
#include "kits/userPoint.h"
#include "vectorTypes/vectorBase.h"
#include "storage/dynamicClass.h"

namespace videoPreprocessor {

//================================================================
//
// ProcessTargetKit
//
//================================================================

using ProcessTargetKit = KitCombine<GpuImageConsoleKit, GpuRgbFrameKit, PipeControlKit, AlternativeVersionKit, VerbosityKit, DisplayParamsKit, UserPointKit, SetBusyStatusKit>;

//================================================================
//
// VideoPrepTarget
//
//================================================================

struct VideoPrepTarget
{
    virtual void inspectProcess(ProcessInspector& inspector) =0;
    virtual stdbool process(stdPars(ProcessTargetKit)) =0;
};

//================================================================
//
// ReallocKit
// ProcessKit
//
//================================================================

using ReallocKit = KitCombine<ModuleReallocKit, GpuAppExecKit, AtCommonKit>;

using ProcessKit = KitCombine<CpuFuncKit, ErrorLogExKit, MsgLogsKit, OverlayTakeoverKit, PipeControlKit, TimerKit, VerbosityKit, 
    UserPointKit, SetBusyStatusKit, GpuAppExecKit, FileToolsKit, MallocKit, AtProcessKit, FrameAdvanceKit>;

//================================================================
//
// VideoPreprocessor
//
//================================================================

class VideoPreprocessor
{

public:

    VideoPreprocessor();
    ~VideoPreprocessor();

    void serialize(const ModuleSerializeKit& kit);
    void setFrameSize(const Point<Space>& frameSize);

    bool reallocValid() const;
    stdbool realloc(stdPars(ReallocKit));

    Point<Space> outputFrameSize() const;
    stdbool process(VideoPrepTarget& target, stdPars(ProcessKit));

private:

    DynamicClass<class VideoPreprocessorImpl> instance;

};

//----------------------------------------------------------------

}
