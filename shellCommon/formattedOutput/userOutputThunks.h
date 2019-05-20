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
// ErrorLogExThunk
//
//================================================================

class ErrorLogExThunk : public ErrorLogEx
{

public:

    bool isThreadProtected() const;

    bool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars);

public:

    inline ErrorLogExThunk()
        {}

    inline ErrorLogExThunk(MsgLog& msgLog)
        : msgLog(&msgLog) {}

    inline void setup(MsgLog& msgLog)
        {this->msgLog = &msgLog;}

    inline void reset()
        {this->msgLog = 0;}

private:

    MsgLog* msgLog = 0;

};
