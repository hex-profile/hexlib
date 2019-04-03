#pragma once

#include "gpuModuleHeader.h"
#include "vectorTypes/vectorBase.h"
#include "atInterface/atInterfaceKit.h"
#include "kits/userPoint.h"
#include "interfaces/fileTools.h"
#include "allocation/mallocKit.h"
#include "kits/displayParamsKit.h"
#include "kits/alternativeVersionKit.h"
#include "atAssembly/frameAdvanceKit.h"
#include "interfaces/threadManagerKit.h"

namespace videoPreprocessor {

//================================================================
//
// GpuRgbFrameKit
//
//================================================================

KIT_CREATE1(GpuRgbFrameKit, GpuMatrix<const uint8_x4>, gpuRgbFrame);

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
    virtual bool process(stdPars(ProcessTargetKit)) =0;
};

//================================================================
//
// ReallocKit
// ProcessKit
//
//================================================================

KIT_COMBINE3(ReallocKit, ModuleReallocKit, GpuAppExecKit, AtCommonKit);
KIT_COMBINE7(ProcessKit, ModuleProcessKit, GpuAppExecKit, FileToolsKit, MallocKit, AtProcessKit, FrameAdvanceKit, ThreadManagerKit);

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
    bool realloc(const Point<Space>& frameSize, stdPars(ReallocKit));

    Point<Space> outputFrameSize() const;
    bool processEntry(VideoPrepTarget& target, stdPars(ProcessKit));

private:

    StaticClass<class VideoPreprocessorImpl, 1 << 15> instance;

};

//----------------------------------------------------------------

}
