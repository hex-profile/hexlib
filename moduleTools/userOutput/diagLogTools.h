#pragma once

#include "diagLog/diagLog.h"
#include "userOutput/msgLog.h"

//================================================================
//
// DiagLogByMsgLog
//
//================================================================

class DiagLogByMsgLog : public DiagLog
{

public:

    DiagLogByMsgLog(MsgLog& base)
        : base(base) {}

    bool addMsg(const CharType* msgPtr, MsgKind msgKind) override
        {return base.addMsg(FormatOutputAtom(charArrayFromPtr(msgPtr)), msgKind);}

    bool isThreadProtected() const
        {return base.isThreadProtected();}

    void lock() override
        {return base.lock();}

    void unlock() override
        {return base.unlock();}

private:

    MsgLog& base;

};
