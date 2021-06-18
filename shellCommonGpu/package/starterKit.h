#pragma once

#include "allocation/mallocKit.h"
#include "debugBridge/bridge/debugBridgeKit.h"
#include "errorLog/errorLogKit.h"
#include "interfaces/fileToolsKit.h"
#include "interfaces/threadManagerKit.h"
#include "kits/msgLogsKit.h"
#include "timer/timerKit.h"
#include "userOutput/errorLogExKit.h"
#include "package/packageApi.h"

namespace packageImpl {

using namespace packageApi;

//================================================================
//
// StarterKit
//
//================================================================

using StarterKit = KitCombine<ErrorLogKit, MsgLogsKit, ErrorLogExKit, TimerKit, FileToolsKit, MallocKit, ThreadManagerKit>;

//================================================================
//
// StarterDebugKit
//
//================================================================

KIT_CREATE1(DumpParamsKit, const DumpParams&, dumpParams);

using StarterDebugKit = KitCombine<StarterKit, DebugBridgeKit, DumpParamsKit>;

//----------------------------------------------------------------

}
