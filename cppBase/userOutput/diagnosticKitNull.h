#pragma once

#include "userOutput/diagnosticKit.h"
#include "userOutput/msgLog.h"
#include "errorLog/errorLog.h"
#include "userOutput/errorLogEx.h"

//================================================================
//
// DiagnosticKitNull
//
//================================================================

class DiagnosticKitNull : public DiagnosticKit
{

public:

    inline DiagnosticKitNull()
        : 
        DiagnosticKit(kitCombine(ErrorLogKit(errorLogNull), MsgLogKit(msgLogNull), ErrorLogExKit(errorLogExNull)))
    {
    }

private:

    MsgLogNull msgLogNull;
    ErrorLogNull errorLogNull;
    ErrorLogExNull errorLogExNull;

};
