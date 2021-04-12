#pragma once

#include "debugBridge/bridge/debugBridge.h"
#include "extLib/storage/smartPtr.h"

namespace debugBridgeTest {

using namespace debugBridge;

//================================================================
//
// create
//
//================================================================

UniquePtr<DebugBridge> create();

//----------------------------------------------------------------

}
