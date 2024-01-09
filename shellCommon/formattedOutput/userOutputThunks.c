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
// ErrorLogByMsgLog::addErrorTrace
//
//================================================================

void ErrorLogByMsgLog::addErrorTrace(const CharType* message, TRACE_PARAMS(trace))
{
    if (msgLog)
    {
        MessageBlockGuard guard(*msgLog);

        printMsg(*msgLog, STR("%0"), charArrayFromPtr(message), msgErr);

        int32 depth = 0;

        for (const TraceScope* p = &trace; p != 0; p = p->prev, ++depth)
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
// MsgLogExByMsgLog::addMsgTrace
//
//================================================================

stdbool MsgLogExByMsgLog::addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdParsNull)
{
    const TraceScope* p = &trace;

    if (msgLog)
    {
        MessageBlockGuard guard(*msgLog);

        require(printMsg(*msgLog, STR("%0"), v, msgKind));
        require(printMsg(*msgLog, STR("    %0:"), charArrayFromPtr(p->location), msgKind));

        int32 depth = 0;

        for (p = p->prev; p != 0; p = p->prev, ++depth)
        {
            if (depth < maxTraceDepth)
            {
                require(printMsg(*msgLog, STR("    %0:"), charArrayFromPtr(p->location), msgKind));
            }
            else
            {
                require(printMsg(*msgLog, STR("    ... and so on"), msgKind));
                break;
            }
        }
    }

    returnTrue;
}
