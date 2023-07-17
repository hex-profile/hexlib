#pragma once

#include <stddef.h>

#include "extLib/userOutput/msgKind.h"
#include "extLib/types/charType.h"

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
