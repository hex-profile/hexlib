#pragma once

#include "cfg/cfgSerialization.h"
#include "cfgVars/cfgSerializeImpl/cfgTemporary.h"
#include "cfgVars/cfgTree/cfgTree.h"
#include "stdFunc/stdFunc.h"
#include "userOutput/diagnosticKit.h"

namespace cfgSerializeImpl {

using namespace cfgVarsImpl;

//================================================================
//
// Kit
//
//================================================================

using Kit = DiagnosticKit;

//================================================================
//
// saveVarsToTree
//
//================================================================

struct SaveVarsToTreeArgs
{
    CfgSerialization& serialization;
    CfgTree& cfgTree;
    CfgTemporary& temp;
    bool const saveOnlyUnsyncedVars;
    bool const updateSyncedFlag;
    bool const debugPrint;
};

stdbool saveVarsToTree(const SaveVarsToTreeArgs& args, stdPars(Kit));

//================================================================
//
// loadVarsFromTree
//
//================================================================

struct LoadVarsFromTreeArgs
{
    CfgSerialization& serialization;
    CfgTree& cfgTree;
    CfgTemporary& temp;
    bool const loadOnlyUnsyncedVars;
    bool const updateSyncedFlag;
};

stdbool loadVarsFromTree(const LoadVarsFromTreeArgs& args, stdPars(Kit));

//----------------------------------------------------------------

}
