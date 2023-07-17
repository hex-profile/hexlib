#pragma once

#include "channels/configService/configService.h"
#include "channels/guiService/guiService.h"
#include "cfg/cfgSerialization.h"
#include "stdFunc/stdFunc.h"
#include "storage/smartPtr.h"
#include "userOutput/diagnosticKit.h"
#include "cfgVars/types/charTypes.h"
#include "simpleString/simpleString.h"
#include "timer/timerKit.h"

namespace configKeeper {

//================================================================
//
// ConfigKeeper
//
//================================================================

struct ConfigKeeper
{
    static UniquePtr<ConfigKeeper> create();
    virtual ~ConfigKeeper() {}

    //----------------------------------------------------------------
    //
    // Config.
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
        CfgSerialization& serialization;
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
        configService::ServerApi& configService;
        guiService::ClientApi& guiService;
    };

    virtual void run(const RunArgs& args) =0;


};

//----------------------------------------------------------------

}

using configKeeper::ConfigKeeper;
