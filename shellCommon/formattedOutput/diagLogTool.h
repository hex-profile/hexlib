#pragma once

#include "diagLog/diagLog.h"
#include "storage/disposableObject.h"
#include "userOutput/diagnosticKit.h"
#include "userOutput/msgLog.h"
#include "formattedOutput/userOutputThunks.h"

//================================================================
//
// DiagLogMsgLog
//
//================================================================

class DiagLogMsgLog : public MsgLog
{

public:

    inline DiagLogMsgLog(DiagLog* output = nullptr)
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
// DiagLogTool
//
//================================================================

class DiagLogTool
{

public:

    DiagLogTool() = default;

    DiagLogTool(DiagLog* diagLog)
    {
        create(diagLog);
    }

    void create(DiagLog* diagLog) // may be nullptr
    {
        msgLog.create(diagLog);
        errorLog.create(*msgLog);
        errorLogEx.create(*msgLog);
    }

    void destroy()
    {
        msgLog.destroy();
        errorLog.destroy();
        errorLogEx.destroy();
    }

    DiagnosticKit kit()
    {
        return kitCombine(ErrorLogKit(*errorLog), MsgLogKit(*msgLog), ErrorLogExKit(*errorLogEx));
    }

private:

    DisposableObject<DiagLogMsgLog> msgLog;
    DisposableObject<ErrorLogThunk> errorLog;
    DisposableObject<ErrorLogExThunk> errorLogEx;

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

    void add(const CharType* msgPtr, MsgKind msgKind) override
        {base.addMsg(FormatOutputAtom(charArrayFromPtr(msgPtr)), msgKind);}

    bool isThreadProtected() const
        {return base.isThreadProtected();}

    void lock() override
        {return base.lock();}

    void unlock() override
        {return base.unlock();}

private:

    MsgLog& base;

};
