#pragma once

#include "formatting/formatOutputAtom.h"
#include "userOutput/msgLogKit.h"
#include "userOutput/msgKind.h"

//================================================================
//
// MsgLogLocking
//
// An interface to make continuous message blocks
// in a multithreaded environment.
//
// Allows nested locking.
//
//================================================================

struct MsgLogLocking
{
    virtual bool isThreadProtected() const =0;

    virtual void lock() =0;

    virtual void unlock() =0;
};

//================================================================
//
// MsgLogGuard
//
//================================================================

class MsgLogGuard
{

public:

    MsgLogGuard(MsgLogLocking& locking)
        : locking(locking) {locking.lock();}

    ~MsgLogGuard()
        {locking.unlock();}

private:

    MsgLogLocking& locking;

};

//================================================================
//
// MsgLog
//
// Message log interface.
//
// In application code, use the high-level printf-like functions.
//
//================================================================

struct MsgLog : public MsgLogLocking
{
    // Add message to the log.
    virtual bool addMsg(const FormatOutputAtom& v, MsgKind msgKind) =0;

    // Clear log (if supported).
    virtual bool clear() =0;

    // Update log view (if supported).
    virtual bool update() =0;
};

//================================================================
//
// MsgLogNull
//
//================================================================

struct MsgLogNull : public MsgLog
{
    virtual bool addMsg(const FormatOutputAtom& v, MsgKind msgKind)
        {return true;}

    virtual bool clear()
        {return true;}

    virtual bool update()
        {return true;}

    virtual bool isThreadProtected() const
        {return true;}

    virtual void lock()
        {}

    virtual void unlock()
        {}
};
