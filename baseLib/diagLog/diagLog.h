#pragma once

#include <stddef.h>

#include "types/charType.h"
#include "diagLog/msgKind.h"

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
    // Add a message to a log.
    //

    virtual void add(const CharType* msgPtr, MsgKind msgKind) =0;

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
// DiagLogNull
//
//================================================================

struct DiagLogNull : public DiagLog
{
    void add(const CharType* msgPtr, MsgKind msgKind) override
        {}

    bool isThreadProtected() const override
        {return true;}

    void lock() override
        {}

    void unlock() override
        {}
};
