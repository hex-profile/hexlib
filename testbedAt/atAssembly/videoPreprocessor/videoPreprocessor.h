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
#include "kits/userPoint.h"
#include "vectorTypes/vectorBase.h"

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
KIT_COMBINE7(ProcessKit, ModuleBaseProcessKit, GpuAppExecKit, FileToolsKit, MallocKit, AtProcessKit, FrameAdvanceKit, ThreadManagerKit);

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

    bool reallocValid() const;
    stdbool realloc(const Point<Space>& frameSize, stdPars(ReallocKit));

    Point<Space> outputFrameSize() const;
    stdbool processEntry(VideoPrepTarget& target, stdPars(ProcessKit));

private:

    StaticClass<class VideoPreprocessorImpl, 1 << 15> instance;

};

//----------------------------------------------------------------

}
