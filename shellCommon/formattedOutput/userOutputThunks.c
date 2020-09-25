#include "userOutputThunks.h"

#include "userOutput/printMsg.h"
#include "userOutput/msgLog.h"

//================================================================
//
// MessageBlockGuard
//
//================================================================

class MessageBlockGuard
{

public:

    MessageBlockGuard(MsgLogThreading& locking)
        : locking(locking) {locking.lock();}

    ~MessageBlockGuard()
        {locking.unlock();}

private:

    MsgLogThreading& locking;

};

//================================================================
//
// maxTraceDepth
//
//================================================================

static const int32 maxTraceDepth = 8;

//================================================================
//
// ErrorLogByMsgLog::isThreadProtected
//
//================================================================

bool ErrorLogByMsgLog::isThreadProtected() const 
{
    return msgLog ? msgLog->isThreadProtected() : true;
}

//================================================================
//
// ErrorLogByMsgLog::addErrorSimple
//
//================================================================

void ErrorLogByMsgLog::addErrorSimple(const CharType* message)
{
    if (msgLog)
        printMsg(*msgLog, STR("%0"), charArrayFromPtr(message), msgErr);
}

//================================================================
//
// ErrorLogByMsgLog::addErrorTrace
//
//================================================================

void ErrorLogByMsgLog::addErrorTrace(const CharType* message, TRACE_PARAMS(trace))
{
    TRACE_REASSEMBLE(trace);

    if (msgLog)
    {
        MessageBlockGuard guard(*msgLog);

        printMsg(*msgLog, STR("%0"), charArrayFromPtr(message), msgErr);

        int32 depth = 0;

        for (const TraceScope* p = &TRACE_SCOPE(trace); p != 0; p = p->prev, ++depth)
        {
            if (depth < maxTraceDepth)
            {
                printMsg(*msgLog, STR("    %0:"), charArrayFromPtr(p->location), msgErr);
            }
            else
            {
                printMsg(*msgLog, STR("    ... and so on"), msgErr);
                break;
            }
        }
    }
}

//================================================================
//
// ErrorLogExByMsgLog::isThreadProtected
//
//================================================================

bool ErrorLogExByMsgLog::isThreadProtected() const
{
    return msgLog ? msgLog->isThreadProtected() : true;
}

//================================================================
//
// ErrorLogExByMsgLog::addMsgTrace
//
//================================================================

bool ErrorLogExByMsgLog::addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars)
{
    const TraceScope* p = &TRACE_SCOPE(stdTraceName);

    if (msgLog)
    {
        MessageBlockGuard guard(*msgLog);

        ensure(printMsg(*msgLog, STR("%0"), v, msgKind));
        ensure(printMsg(*msgLog, STR("    %0:"), charArrayFromPtr(p->location), msgKind));

        int32 depth = 0;

        for (p = p->prev; p != 0; p = p->prev, ++depth)
        {
            if (depth < maxTraceDepth)
            {
                ensure(printMsg(*msgLog, STR("    %0:"), charArrayFromPtr(p->location), msgKind));
            }
            else
            {
                ensure(printMsg(*msgLog, STR("    ... and so on"), msgKind));
                break;
            }
        }
    }

    return true;
}
