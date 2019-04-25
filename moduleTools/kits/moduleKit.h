#pragma once

#include "cfg/cfgInterfaceFwd.h"
#include "cfgTools/overlayTakeover.h"
#include "cpuFuncKit.h"
#include "dataAlloc/memoryAllocatorKit.h"
#include "kits/displayParamsKit.h"
#include "kits/alternativeVersionKit.h"
#include "kits/msgLogsKit.h"
#include "kits/userPoint.h"
#include "timer/timerKit.h"
#include "userOutput/errorLogExKit.h"

//================================================================
//
// maxRollbackCount
//
//================================================================

static const Space maxRollbackCount = 8;

//================================================================
//
// PipeControl
//
//================================================================

KIT_CREATE2(
    PipeControl,

    // The number of input frames to roll back BEFORE processing
    Space, rollbackFrames,

    // Re-roll realization
    bool, randomize
);

//----------------------------------------------------------------

KIT_CREATE1(PipeControlKit, const PipeControl&, pipeControl);

//================================================================
//
// OutputLevelKit
//
//================================================================

enum OutputLevel {OUTPUT_NONE, OUTPUT_RENDER, OUTPUT_ENABLED};

KIT_CREATE1(OutputLevelKit, OutputLevel, outputLevel);

//================================================================
//
// ModuleReallocKit
// ModuleProcessKit
//
//================================================================

KIT_COMBINE3(ModuleReallocKit, CpuFuncKit, ErrorLogExKit, MsgLogsKit);

// ```
KIT_COMBINE8(ModuleBaseProcessKit, CpuFuncKit, ErrorLogExKit, MsgLogsKit, OverlayTakeoverKit, PipeControlKit, TimerKit, OutputLevelKit, UserPointKit);
KIT_COMBINE3(ModuleProcessKit, ModuleBaseProcessKit, AlternativeVersionKit, DisplayParamsKit);

//================================================================
//
// ModuleSerializeKit
//
//================================================================

KIT_COMBINE2(ModuleSerializeKit, CfgSerializeKit, OverlayTakeoverKit);

#define CFG_NAMESPACE(name) \
    CFG_NAMESPACE_EX(CT(name))

//================================================================
//
// MODULE_OUTPUT_ENABLED
//
//================================================================

#define MODULE_OUTPUT_ENABLED \
    (kit.dataProcessing && kit.outputLevel >= OUTPUT_ENABLED)
