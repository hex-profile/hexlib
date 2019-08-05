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
// MAX_TRACE_DEPTH
//
//================================================================

static const int32 MAX_TRACE_DEPTH = 6;
static const MsgKind errorMsgKind = msgErr;

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
        printMsg(*msgLog, STR("%0"), charArrayFromPtr(message), errorMsgKind);
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

        printMsg(*msgLog, STR("%0"), charArrayFromPtr(message), errorMsgKind);

        int32 depth = 0;

        for (const TraceScope* p = &TRACE_SCOPE(trace); p != 0; p = p->prev)
        {
            if_not (depth < MAX_TRACE_DEPTH)
            {
                printMsg(*msgLog, STR("    ... and so on"), errorMsgKind);
                break;
            }

            printMsg(*msgLog, STR("    %0: called from"), charArrayFromPtr(p->location), errorMsgKind);
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
            if_not (depth < MAX_TRACE_DEPTH)
            {
                ensure(printMsg(*msgLog, STR("    ... and so on"), msgKind));
                break;
            }

            ensure(printMsg(*msgLog, STR("    %0:"), charArrayFromPtr(p->location), msgKind));
        }
    }

    return true;
}
