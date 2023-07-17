#pragma once

#include "allocation/mallocKit.h"
#include "debugBridge/bridge/debugBridgeKit.h"
#include "errorLog/errorLogKit.h"
#include "kits/msgLogsKit.h"
#include "timer/timerKit.h"
#include "userOutput/msgLogExKit.h"
#include "package/packageApi.h"

namespace packageImpl {

using namespace packageApi;

//================================================================
//
// StarterKit
//
//================================================================

using StarterKit = KitCombine<ErrorLogKit, MsgLogsKit, MsgLogExKit, TimerKit, MallocKit>;

//================================================================
//
// StarterDebugKit
//
//================================================================

KIT_CREATE(DumpParamsKit, const DumpParams&, dumpParams);

using StarterDebugKit = KitCombine<StarterKit, DebugBridgeKit, DumpParamsKit>;

//----------------------------------------------------------------

}
