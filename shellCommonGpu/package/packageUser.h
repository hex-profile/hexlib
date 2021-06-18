#pragma once

#include "formattedOutput/diagLogTool.h"
#include "debugBridge/bridge/debugBridgeFwd.h"
#include "package/packageApi.h"

namespace packageUser {

using namespace packageApi;

//================================================================
//
// PackageDebugKitMaker
//
//================================================================

struct PackageDebugKitMaker
{
    PackageDebugKitMaker(const MsgLogKit& kit, DebugBridge& debugBridge, const DumpParams& dumpParams)
        : 
        diagLogImpl{kit.msgLog},
        dumpParams{dumpParams},
        kit{&diagLogImpl, &debugBridge, dumpParams}
    {
    }

    DiagLogByMsgLog diagLogImpl;
    DumpParams const dumpParams;
    PackageDebugKit const kit;
};

//----------------------------------------------------------------

}
