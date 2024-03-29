#pragma once

#include "cfgVars/types/charTypes.h"
#include "cfg/cfgSerialization.h"
#include "cfgVars/types/stringReceiver.h"
#include "cfgVars/cfgTree/cfgTree.h"
#include "stdFunc/stdFunc.h"
#include "storage/smartPtr.h"
#include "userOutput/diagnosticKit.h"

namespace cfgOperations {

using namespace cfgVarsImpl;

//================================================================
//
// Kit.
//
//================================================================

using Kit = DiagnosticKit;

//================================================================
//
// SaveVarsOptions
// LoadVarsOptions
//
//================================================================

struct SaveVarsOptions
{
    bool const saveOnlyUnsyncedVars;
    bool const updateSyncedFlag;
};

struct LoadVarsOptions
{
    bool const loadOnlyUnsyncedVars;
    bool const updateSyncedFlag;
};

//================================================================
//
// CfgOperations
//
//================================================================

struct CfgOperations
{
    static UniquePtr<CfgOperations> create();
    virtual ~CfgOperations() {}

    virtual void dealloc() =0;

    //
    // File I/O.
    //

    virtual stdbool loadFromFile(CfgTree& memory, const Char* filename, bool trackDataChange, stdPars(Kit)) =0;
    virtual stdbool saveToFile(CfgTree& memory, const Char* filename, stdPars(Kit)) =0;

    //
    // String I/O.
    //

    virtual stdbool loadFromString(CfgTree& memory, const StringRef& str, stdPars(Kit)) =0;
    virtual stdbool saveToString(CfgTree& memory, StringReceiver& receiver, stdPars(Kit)) =0;

    //
    // Serialization support.
    //

    virtual stdbool saveVars(CfgTree& memory, CfgSerialization& serialization, const SaveVarsOptions& options, stdPars(Kit)) =0;
    virtual stdbool loadVars(CfgTree& memory, CfgSerialization& serialization, const LoadVarsOptions& options, stdPars(Kit)) =0;

};

//----------------------------------------------------------------

}

using cfgOperations::CfgOperations;
