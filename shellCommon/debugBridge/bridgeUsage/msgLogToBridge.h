#pragma once

#include "extLib/userOutput/diagLog.h"
#include "formatting/messageFormatter.h"
#include "userOutput/msgLog.h"
#include "debugBridge/bridge/debugBridge.h"

//================================================================
//
// MsgLogToDiagLogAndBridge
//
//================================================================

class MsgLogToDiagLogAndBridge : public MsgLog
{

public:

    inline MsgLogToDiagLogAndBridge
    (
        MessageFormatter& formatter,
        bool const flushEveryMessage,
        DiagLog& diagLog,
        bool bridgeActive,
        debugBridge::MessageConsole& bridgeLog
    )
        : 
        formatter(formatter),
        flushEveryMessage(flushEveryMessage),
        diagLog(diagLog),
        bridgeActive(bridgeActive),
        bridgeLog(bridgeLog) 
    {
    }

public:

    bool isThreadProtected() const
        {return diagLog.isThreadProtected();}

    void lock()
        {return diagLog.lock();}

    void unlock()
        {return diagLog.unlock();}

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind);

    bool clear()
        {return diagLog.clear();}

    bool update()
        {return diagLog.update();}

private:

    MessageFormatter& formatter;
    bool const flushEveryMessage;
    DiagLog& diagLog;
    bool const bridgeActive;
    debugBridge::MessageConsole& bridgeLog;

};
