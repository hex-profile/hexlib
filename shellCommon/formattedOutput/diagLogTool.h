#pragma once

#include "extLib/userOutput/diagLog.h"
#include "formattedOutput/userOutputThunks.h"
#include "storage/optionalObject.h"
#include "userOutput/diagnosticKit.h"
#include "userOutput/msgLog.h"
#include "formatting/messageFormatter.h"

//================================================================
//
// MsgLogByDiagLog
//
//================================================================

class MsgLogByDiagLog : public MsgLog
{

public:

    inline MsgLogByDiagLog(DiagLog& base, MessageFormatter& formatter)
        : base(base), formatter(formatter) {}

public:

    bool isThreadProtected() const
        {return base.isThreadProtected();}

    void lock()
        {return base.lock();}

    void unlock()
        {return base.unlock();}

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind);

    bool clear()
        {return base.clear();}

    bool update()
        {return base.update();}

private:

    DiagLog& base;
    MessageFormatter& formatter;

};

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

    virtual bool isThreadProtected() const
        {return base.isThreadProtected();}

    virtual void lock() override
        {return base.lock();}

    virtual void unlock() override
        {return base.unlock();}

    virtual bool addMsg(const CharType* msgPtr, MsgKind msgKind) override
        {return base.addMsg(FormatOutputAtom(charArrayFromPtr(msgPtr)), msgKind);}

    virtual bool clear()
        {return base.clear();}

    virtual bool update()
        {return base.update();}

private:

    MsgLog& base;

};
