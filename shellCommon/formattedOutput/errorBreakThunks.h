#pragma once

#include "userOutput/msgLog.h"
#include "errorLog/debugBreak.h"
#include "compileTools/classContext.h"
#include "errorLog/errorLog.h"
#include "userOutput/printMsgTrace.h"

//================================================================
//
// MsgLogBreakShell
//
//================================================================

class MsgLogBreakShell : public MsgLog
{

public:

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind)
    {
        bool ok = base.addMsg(v, msgKind);

        if (msgKind == msgErr && debugBreakOnErrors)
            DEBUG_BREAK_INLINE();

        return ok;
    }

    bool clear()
        {return base.clear();}

    bool update()
        {return base.update();}

    void lock()
        {return base.lock();}

    void unlock()
        {return base.unlock();}

private:

    CLASS_CONTEXT(MsgLogBreakShell, ((MsgLog&, base)) ((bool, debugBreakOnErrors)))

};

//================================================================
//
// ErrorLogBreakShell
//
//================================================================

class ErrorLogBreakShell : public ErrorLog
{

public:

    void addErrorTrace(const CharType* message, TRACE_PARAMS(trace))
    {
        base.addErrorTrace(message, TRACE_PASSTHRU(trace));

        if (debugBreakOnErrors)
            DEBUG_BREAK_INLINE();
    }

private:

    CLASS_CONTEXT(ErrorLogBreakShell, ((ErrorLog&, base)) ((bool, debugBreakOnErrors)))

};

//================================================================
//
// MsgLogExBreakShell
//
//================================================================

class MsgLogExBreakShell : public MsgLogEx
{

public:

    bool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars)
    {
        bool ok = base.addMsgTrace(v, msgKind, stdNullPassThru);

        if (msgKind == msgErr && debugBreakOnErrors)
            DEBUG_BREAK_INLINE();

        return ok;
    }

private:

    CLASS_CONTEXT(MsgLogExBreakShell, ((MsgLogEx&, base)) ((bool, debugBreakOnErrors)))

};
