#pragma once

#include "extLib/diagLog/diagLog.h"
#include "formattedOutput/userOutputThunks.h"
#include "storage/disposableObject.h"
#include "userOutput/diagnosticKit.h"
#include "userOutput/msgLog.h"

//================================================================
//
// MsgLogByDiagLog
//
//================================================================

class MsgLogByDiagLog : public MsgLog
{

public:

    inline MsgLogByDiagLog(DiagLog* output = nullptr)
        : output(output) {}

    inline void setup(DiagLog* output)
        {this->output = output;}

    inline void reset()
        {this->output = nullptr;}

public:

    bool isThreadProtected() const
        {return !output ? true : output->isThreadProtected();}

    void lock()
        {if (output) output->lock();}

    void unlock()
        {if (output) output->unlock();}

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind);

    bool clear()
        {return true;}

    bool update()
        {return true;}

private:

    DiagLog* output = nullptr;

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
