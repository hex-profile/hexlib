#include "userOutputThunks.h"

#include "userOutput/printMsg.h"
#include "userOutput/msgLog.h"

//================================================================
//
// MAX_TRACE_DEPTH
//
//================================================================

static const int32 MAX_TRACE_DEPTH = 6;
static const MsgKind errorMsgKind = msgErr;

//================================================================
//
// ErrorLogThunk::isThreadProtected
//
//================================================================

bool ErrorLogThunk::isThreadProtected() const 
{
    return msgLog ? msgLog->isThreadProtected() : true;
}

//================================================================
//
// ErrorLogThunk::addErrorSimple
//
//================================================================

void ErrorLogThunk::addErrorSimple(const CharType* message)
{
    if (msgLog)
        printMsg(*msgLog, STR("%0"), charArrayFromPtr(message), errorMsgKind);
}

//================================================================
//
// ErrorLogThunk::addErrorTrace
//
//================================================================

void ErrorLogThunk::addErrorTrace(const CharType* message, TRACE_PARAMS(trace))
{
    TRACE_REASSEMBLE(trace);

    if (msgLog)
    {
        MsgLogGuard guard(*msgLog);

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
// ErrorLogExThunk::isThreadProtected
//
//================================================================

bool ErrorLogExThunk::isThreadProtected() const
{
    return msgLog ? msgLog->isThreadProtected() : true;
}

//================================================================
//
// ErrorLogExThunk::addMsgTrace
//
//================================================================

bool ErrorLogExThunk::addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars)
{
    stdNullBegin;

    const TraceScope* p = &TRACE_SCOPE(stdTraceName);

    if (msgLog)
    {
        MsgLogGuard guard(*msgLog);

        require(printMsg(*msgLog, STR("%0"), v, msgKind));
        require(printMsg(*msgLog, STR("    %0:"), charArrayFromPtr(p->location), msgKind));

        int32 depth = 0;

        for (p = p->prev; p != 0; p = p->prev, ++depth)
        {
            if_not (depth < MAX_TRACE_DEPTH)
            {
                require(printMsg(*msgLog, STR("    ... and so on"), msgKind));
                break;
            }

            require(printMsg(*msgLog, STR("    %0:"), charArrayFromPtr(p->location), msgKind));
        }
    }

    stdEnd;
}
