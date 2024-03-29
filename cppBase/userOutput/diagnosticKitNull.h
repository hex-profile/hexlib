#pragma once

#include "userOutput/diagnosticKit.h"
#include "userOutput/msgLog.h"
#include "errorLog/errorLog.h"
#include "userOutput/printMsgTrace.h"
#include "formatting/messageFormatter.h"

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
        DiagnosticKit(kitCombine(MessageFormatterKit(formatterNull), ErrorLogKit(errorLogNull), MsgLogKit(msgLogNull), MsgLogExKit(msgLogExNull)))
    {
    }

private:

    MessageFormatterNull formatterNull;
    MsgLogNull msgLogNull;
    ErrorLogNull errorLogNull;
    MsgLogExNull msgLogExNull;

};
