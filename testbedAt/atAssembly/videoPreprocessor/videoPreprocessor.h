#pragma once

#include "allocation/mallocKit.h"
#include "atAssembly/frameAdvanceKit.h"
#include "atInterface/atInterfaceKit.h"
#include "gpuModuleHeader.h"
#include "interfaces/fileTools.h"
#include "interfaces/threadManagerKit.h"
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

KIT_COMBINE7(ProcessTargetKit, GpuImageConsoleKit, GpuRgbFrameKit, PipeControlKit, AlternativeVersionKit, OutputLevelKit, DisplayParamsKit, UserPointKit);

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

KIT_COMBINE3(ReallocKit, ModuleReallocKit, GpuAppExecKit, AtCommonKit);

KIT_COMBINE8(ProcessBaseKit_, CpuFuncKit, ErrorLogExKit, MsgLogsKit, OverlayTakeoverKit, PipeControlKit, TimerKit, OutputLevelKit, UserPointKit);
KIT_COMBINE7(ProcessKit, ProcessBaseKit_, GpuAppExecKit, FileToolsKit, MallocKit, AtProcessKit, FrameAdvanceKit, ThreadManagerKit);

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
