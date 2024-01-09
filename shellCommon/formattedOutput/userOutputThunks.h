#pragma once

#include "errorLog/errorLog.h"
#include "userOutput/msgLog.h"
#include "userOutput/printMsgTrace.h"

//================================================================
//
// ErrorLogByMsgLog
//
//================================================================

class ErrorLogByMsgLog : public ErrorLog
{

public:

    inline ErrorLogByMsgLog(MsgLog& msgLog)
        : msgLog(&msgLog) {}

    void addErrorTrace(const CharType* message, TRACE_PARAMS(trace));

public:

    inline void setup(MsgLog& msgLog)
        {this->msgLog = &msgLog;}

    inline void reset()
        {msgLog = nullptr;}

private:

    MsgLog* msgLog = nullptr;

};

//================================================================
//
// MsgLogExByMsgLog
//
//================================================================

class MsgLogExByMsgLog : public MsgLogEx
{

public:

    stdbool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdParsNull);

public:

    inline MsgLogExByMsgLog()
        {}

    inline MsgLogExByMsgLog(MsgLog& msgLog)
        : msgLog(&msgLog) {}

    inline void setup(MsgLog& msgLog)
        {this->msgLog = &msgLog;}

    inline void reset()
        {this->msgLog = 0;}

private:

    MsgLog* msgLog = 0;

};
