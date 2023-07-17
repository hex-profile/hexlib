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
#include "kits/userPointKit.h"
#include "timer/timerKit.h"
#include "userOutput/msgLogExKit.h"

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

struct PipeControl
{
    // The number of input frames to roll back BEFORE processing
    Space rollbackFrames;

    // Re-roll realization
    bool randomize;
};

//----------------------------------------------------------------

KIT_CREATE(PipeControlKit, const PipeControl&, pipeControl);

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

KIT_CREATE(VerbosityKit, Verbosity, verbosity);

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

using ModuleReallocKit = KitCombine<CpuFuncKit, MsgLogExKit, MsgLogsKit>;

using ModuleProcessKit = KitCombine<CpuFuncKit, MsgLogExKit, MsgLogsKit,
    PipeControlKit, TimerKit, VerbosityKit, UserPointKit, SetBusyStatusKit,
    AlternativeVersionKit, DisplayParamsKit>;

//================================================================
//
// ModuleSerializeKit
//
//================================================================

using ModuleSerializeKit = KitCombine<CfgSerializeKit, OverlayTakeoverKit>;
