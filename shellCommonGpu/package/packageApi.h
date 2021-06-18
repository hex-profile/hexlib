#pragma once

#include "debugBridge/bridge/debugBridgeFwd.h"
#include "extLib/userOutput/diagLog.h"
#include "extLib/types/intBase.h"

namespace packageApi {

//================================================================
//
// DumpParams
//
//================================================================

struct DumpParams
{
    // Numerical prefix for the current dump.
    uint32_t dumpIndex = 0;

    DumpParams() =default;

    DumpParams(uint32_t dumpIndex)
        : dumpIndex{dumpIndex} {}
};

//================================================================
//
// PackageDebugKit
//
//================================================================

struct PackageDebugKit
{
    // Use nullptr to disable.
    DiagLog* const log;

    // Use nullptr to disable.
    DebugBridge* const debugBridge;

    DumpParams dumpParams;

    PackageDebugKit(DiagLog* log, DebugBridge* debugBridge, const DumpParams& dumpParams)
        : log{log}, debugBridge{debugBridge}, dumpParams{dumpParams} {}
};

//================================================================
//
// PackageKit
//
//================================================================

struct PackageKit
{
    // Use nullptr to disable.
    DiagLog* const log;

    PackageKit(DiagLog* log)
        : log{log} {}

    PackageKit(const PackageDebugKit& kit)
        : log{kit.log} {}
};

//----------------------------------------------------------------

}
