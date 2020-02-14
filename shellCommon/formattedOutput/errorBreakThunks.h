#pragma once

#include "userOutput/msgLog.h"
#include "errorLog/debugBreak.h"
#include "compileTools/classContext.h"
#include "errorLog/errorLog.h"
#include "userOutput/errorLogEx.h"

//================================================================
//
// MsgLogBreakShell
//
//================================================================

class MsgLogBreakShell : public MsgLog
{

public:

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind) override
    {
        bool ok = base.addMsg(v, msgKind);

        if (msgKind == msgErr && debugBreakOnErrors)
            DEBUG_BREAK_INLINE();

        return ok;
    }

    bool clear() override
        {return base.clear();}

    bool update() override
        {return base.update();}
    
    bool isThreadProtected() const override
        {return base.isThreadProtected();}

    void lock() override
        {return base.lock();}

    void unlock() override
        {return base.unlock();}

private:
    
    CLASS_CONTEXT(MsgLogBreakShell, ((MsgLog&, base)) ((bool, debugBreakOnErrors)));

};

//================================================================
//
// ErrorLogBreakShell
//
//================================================================

class ErrorLogBreakShell : public ErrorLog
{

public:

    bool isThreadProtected() const override
    {
        return base.isThreadProtected();
    }

    void addErrorSimple(const CharType* message) override
    {
        base.addErrorSimple(message);

        if (debugBreakOnErrors)
            DEBUG_BREAK_INLINE();
    }

    void addErrorTrace(const CharType* message, TRACE_PARAMS(trace)) override
    {
        base.addErrorTrace(message, TRACE_PASSTHRU(trace));

        if (debugBreakOnErrors)
            DEBUG_BREAK_INLINE();
    }

private:
    
    CLASS_CONTEXT(ErrorLogBreakShell, ((ErrorLog&, base)) ((bool, debugBreakOnErrors)));

};

//================================================================
//
// ErrorLogExBreakShell
//
//================================================================

class ErrorLogExBreakShell : public ErrorLogEx
{

public:

    bool isThreadProtected() const override
    {
        return base.isThreadProtected();
    }

    bool addMsgTrace(const FormatOutputAtom& v, MsgKind msgKind, stdNullPars)
    {
        bool ok = base.addMsgTrace(v, msgKind, stdNullPassThru);

        if (msgKind == msgErr && debugBreakOnErrors)
            DEBUG_BREAK_INLINE();

        return ok;
    }

private:
    
    CLASS_CONTEXT(ErrorLogExBreakShell, ((ErrorLogEx&, base)) ((bool, debugBreakOnErrors)));

};
