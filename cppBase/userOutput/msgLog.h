#pragma once

#include "formatting/formatOutputAtom.h"
#include "userOutput/msgKind.h"
          
//================================================================
//
// MsgLogThreading
//
// An interface to make continuous message blocks
// in a multithreaded environment.
//
// Should allow nested locking.
//
//================================================================

struct MsgLogThreading
{
    //
    // Can the instance be shared among multiple threads?
    //

    virtual bool isThreadProtected() const =0;

    //
    // An interface to output contiguous message blocks in a multithreaded environment.
    // Used only for pretty printing, so the implementation may be omitted.
    //
    // If implemented, should allow nested thread locking (for example, use std::recursive_mutex).
    //

    virtual void lock() =0;
    virtual void unlock() =0;
};

//================================================================
//
// MsgLogOutput
//
// Message log interface.
//
//================================================================

struct MsgLogOutput
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
// MsgLog
//
// Message log interface.
//
// In application code, use the high-level printf-like functions.
//
//================================================================

struct MsgLog : public MsgLogThreading, public MsgLogOutput
{
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
