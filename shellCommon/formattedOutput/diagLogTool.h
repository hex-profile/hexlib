#pragma once

#include "diagLog/diagLog.h"
#include "storage/disposableObject.h"
#include "userOutput/diagnosticKit.h"
#include "userOutput/msgLog.h"
#include "formattedOutput/userOutputThunks.h"

//================================================================
//
// MsgLogByDiagLogStlFormatting
//
//================================================================

class MsgLogByDiagLogStlFormatting : public MsgLog
{

public:

    inline MsgLogByDiagLogStlFormatting(DiagLog* output = nullptr)
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

    DisposableObject<MsgLogByDiagLogStlFormatting> msgLog;
    DisposableObject<ErrorLogByMsgLog> errorLog;
    DisposableObject<ErrorLogExByMsgLog> errorLogEx;

};
