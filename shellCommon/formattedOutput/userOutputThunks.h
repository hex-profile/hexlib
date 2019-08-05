#pragma once

#include "errorLog/errorLog.h"
#include "userOutput/msgLog.h"
#include "userOutput/errorLogEx.h"

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

    bool isThreadProtected() const override;
    void addErrorSimple(const CharType* message) override;
    void addErrorTrace(const CharType* message, TRACE_PARAMS(trace)) override;

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
// ErrorLogExByMsgLog
//
//================================================================

class ErrorLogExByMsgLog : public ErrorLogEx
{

public:

    bool isThreadProtected() const;

    bool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars);

public:

    inline ErrorLogExByMsgLog()
        {}

    inline ErrorLogExByMsgLog(MsgLog& msgLog)
        : msgLog(&msgLog) {}

    inline void setup(MsgLog& msgLog)
        {this->msgLog = &msgLog;}

    inline void reset()
        {this->msgLog = 0;}

private:

    MsgLog* msgLog = 0;

};
