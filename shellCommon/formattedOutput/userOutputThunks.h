#pragma once

#include "errorLog/errorLog.h"
#include "userOutput/msgLog.h"
#include "userOutput/errorLogEx.h"

//================================================================
//
// ErrorLogThunk
//
//================================================================

class ErrorLogThunk : public ErrorLog
{

public:

    inline ErrorLogThunk(MsgLog& msgLog)
        :
        ErrorLog(isThreadProtected, addErrorSimple, addErrorTrace),
        msgLog(&msgLog)
    {
    }

    inline ErrorLogThunk()
        :
        ErrorLog(isThreadProtected, addErrorSimple, addErrorTrace)
    {
    }

    static bool isThreadProtected(const ErrorLog& self);
    static void addErrorSimple(ErrorLog& self, const CharType* message);
    static void addErrorTrace(ErrorLog& self, const CharType* message, TRACE_PARAMS(trace));

public:

    inline void setup(MsgLog& msgLog)
        {this->msgLog = &msgLog;}

    inline void reset()
        {msgLog = 0;}

private:

    MsgLog* msgLog = nullptr;

};

//================================================================
//
// MsgLogTraceThunk
//
//================================================================

class MsgLogTraceThunk : public ErrorLogEx
{

public:

    bool isThreadProtected() const;

    bool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars);

public:

    inline MsgLogTraceThunk()
        {}

    inline MsgLogTraceThunk(MsgLog& msgLog)
        : msgLog(&msgLog) {}

    inline void setup(MsgLog& msgLog)
        {this->msgLog = &msgLog;}

    inline void reset()
        {this->msgLog = 0;}

private:

    MsgLog* msgLog = 0;

};
