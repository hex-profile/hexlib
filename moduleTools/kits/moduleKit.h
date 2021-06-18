#pragma once

#include "cfg/cfgInterfaceFwd.h"
#include "cfgTools/overlayTakeover.h"
#include "cpuFuncKit.h"
#include "dataAlloc/memoryAllocatorKit.h"
#include "formatting/messageFormatterKit.h"
#include "kits/alternativeVersionKit.h"
#include "kits/displayParamsKit.h"
#include "kits/msgLogsKit.h"
#include "kits/setBusyStatusKit.h"
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
// @@@ Bad interface, remake it as a separate function.
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
// VerbosityKit
//
//================================================================

enum class Verbosity
{
    // Produce nothing.
    Off, 

    // Produce nothing but main fullscreen image.
    Render, 

    // Produce everything.
    On
};

//----------------------------------------------------------------

KIT_CREATE1(VerbosityKit, Verbosity, verbosity);

//----------------------------------------------------------------

template <typename Kit>
sysinline auto verboseOnlyIf(bool condition, const Kit& kit)
    {return kitReplace(kit, VerbosityKit(condition ? kit.verbosity : Verbosity::Off));}

//================================================================
//
// ModuleReallocKit
// ModuleProcessKit
//
//================================================================

using ModuleReallocKit = KitCombine<CpuFuncKit, ErrorLogExKit, MsgLogsKit>;

using ModuleProcessKit = KitCombine<CpuFuncKit, ErrorLogExKit, MsgLogsKit,
    PipeControlKit, TimerKit, VerbosityKit, UserPointKit, SetBusyStatusKit,
    AlternativeVersionKit, DisplayParamsKit>;

//================================================================
//
// ModuleSerializeKit
//
//================================================================

KIT_COMBINE2(ModuleSerializeKit, CfgSerializeKit, OverlayTakeoverKit);
