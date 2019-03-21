#pragma once

#include "userOutput/errorLogExKit.h"
#include "cpuFuncKit.h"
#include "dataAlloc/memoryAllocatorKit.h"
#include "cfg/cfgInterfaceFwd.h"
#include "cfgTools/overlayTakeover.h"
#include "kits/msgLogsKit.h"
#include "kits/userPoint.h"
#include "timer/timerKit.h"

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
KIT_COMBINE8(ModuleProcessKit, CpuFuncKit, ErrorLogExKit, MsgLogsKit, OverlayTakeoverKit, PipeControlKit, TimerKit, OutputLevelKit, UserPointKit);

//================================================================
//
// ModuleSerializeKit
//
//================================================================

KIT_COMBINE2(ModuleSerializeKit, CfgSerializeKit, OverlayTakeoverKit);

#define CFG_NAMESPACE_MODULE(name) \
    CFG_NAMESPACE_EX(name, ModuleSerializeKit)

//================================================================
//
// MODULE_OUTPUT_ENABLED
//
//================================================================

#define MODULE_OUTPUT_ENABLED \
    (kit.dataProcessing && kit.outputLevel >= OUTPUT_ENABLED)

//================================================================
//
// stdBeginModuleProf
// stdEnterModuleProf
//
//================================================================

#define stdBeginModuleProf(userName) \
    stdBeginProfName(MODULE_OUTPUT_ENABLED, userName)

#define stdEnterModuleProf(userName) \
    stdEnterProfName(MODULE_OUTPUT_ENABLED, userName)
