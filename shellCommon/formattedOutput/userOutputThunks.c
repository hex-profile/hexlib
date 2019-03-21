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

bool ErrorLogThunk::isThreadProtected(const ErrorLog& self)
{
    const ErrorLogThunk& that = static_cast<const ErrorLogThunk&>(self);
    return that.msgLog ? that.msgLog->isThreadProtected() : true;
}

//================================================================
//
// ErrorLogThunk::addErrorSimple
//
//================================================================

void ErrorLogThunk::addErrorSimple(ErrorLog& self, const CharType* message)
{
    ErrorLogThunk& that = static_cast<ErrorLogThunk&>(self);

    if (that.msgLog)
        printMsg(*that.msgLog, STR("%0"), charArrayFromPtr(message), errorMsgKind);
}

//================================================================
//
// ErrorLogThunk::addErrorTrace
//
//================================================================

void ErrorLogThunk::addErrorTrace(ErrorLog& self, const CharType* message, TRACE_PARAMS(trace))
{
    TRACE_REASSEMBLE(trace);

    ErrorLogThunk& that = static_cast<ErrorLogThunk&>(self);

    if (that.msgLog)
    {
        MsgLogGuard guard(*that.msgLog);

        printMsg(*that.msgLog, STR("%0"), charArrayFromPtr(message), errorMsgKind);

        int32 depth = 0;

        for (const TraceScope* p = &TRACE_SCOPE(trace); p != 0; p = p->prev)
        {
            if_not (depth < MAX_TRACE_DEPTH)
            {
                printMsg(*that.msgLog, STR("    ... and so on"), errorMsgKind);
                break;
            }

            printMsg(*that.msgLog, STR("    %0: called from"), charArrayFromPtr(p->location), errorMsgKind);
        }
    }
}

//================================================================
//
// MsgLogTraceThunk::isThreadProtected
//
//================================================================

bool MsgLogTraceThunk::isThreadProtected() const
{
    return msgLog ? msgLog->isThreadProtected() : true;
}

//================================================================
//
// MsgLogTraceThunk::addMsgTrace
//
//================================================================

bool MsgLogTraceThunk::addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars)
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
