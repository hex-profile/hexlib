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
// DiagLogTool
//
//================================================================

class DiagLogTool
{

public:

    DiagLogTool(DiagLog* diagLog = nullptr)
        {create(diagLog);}

    void create(DiagLog* diagLog) // may be nullptr
    {
        msgLog.create(diagLog);
        errorLog.create(*msgLog);
        errorLogEx.create(*msgLog);
    }

    DiagnosticKit kit()
    {
        return kitCombine(ErrorLogKit(*errorLog), MsgLogKit(*msgLog), ErrorLogExKit(*errorLogEx));
    }

private:

    DisposableObject<MsgLogByDiagLog> msgLog;
    DisposableObject<ErrorLogByMsgLog> errorLog;
    DisposableObject<ErrorLogExByMsgLog> errorLogEx;

};
