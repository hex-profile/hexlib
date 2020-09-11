#pragma once

#include <stddef.h>

#include "extLib/types/charType.h"

//================================================================
//
// MsgKind
//
// Auxiliary message type specification.
//
//================================================================

#ifndef HEXLIB_MSGKIND
#define HEXLIB_MSGKIND

enum MsgKind {msgInfo, msgWarn, msgErr};

#endif

//================================================================
//
// DiagLog
//
// The simplest diagnostic logging interface.
//
//================================================================

struct DiagLog
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

    // Add message to the log.
    virtual bool addMsg(const CharType* msgStr, MsgKind msgKind) =0;

    // Clear log (if supported).
    virtual bool clear() =0;

    // Update log view (if supported).
    virtual bool update() =0;
};

//================================================================
//
// DiagLogNull
//
//================================================================

struct DiagLogNull : public DiagLog
{
    bool isThreadProtected() const
        {return true;}

    void lock()
        {}

    void unlock()
        {}

    bool addMsg(const CharType* msgStr, MsgKind msgKind)
        {return true;}

    bool clear()
        {return true;}

    bool update()
        {return true;}
};
