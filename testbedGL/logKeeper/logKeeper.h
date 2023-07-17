#pragma once

#include "channels/guiService/guiService.h"
#include "channels/logService/logService.h"
#include "stdFunc/stdFunc.h"
#include "storage/smartPtr.h"
#include "timer/timerKit.h"
#include "userOutput/diagnosticKit.h"
#include "cfg/cfgInterfaceFwd.h"

namespace logKeeper {

//================================================================
//
// LogKeeper
//
//================================================================

struct LogKeeper
{
    static UniquePtr<LogKeeper> create();
    virtual ~LogKeeper() {}

    //----------------------------------------------------------------
    //
    // Log.
    //
    //----------------------------------------------------------------

    virtual void serialize(const CfgSerializeKit& kit) =0;

    //----------------------------------------------------------------
    //
    // Init.
    //
    //----------------------------------------------------------------

    struct InitArgs
    {
        const CharType* baseFilename;
    };

    using InitKit = DiagnosticKit;

    virtual stdbool init(const InitArgs& args, stdPars(InitKit)) =0;

    //----------------------------------------------------------------
    //
    // Run.
    //
    //----------------------------------------------------------------

    struct RunArgs
    {
        logService::ServerApi& logService;
        guiService::ClientApi& guiService;
    };

    virtual void run(const RunArgs& args) =0;
};

//----------------------------------------------------------------

}

using logKeeper::LogKeeper;
