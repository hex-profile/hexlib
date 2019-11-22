#pragma once

#include <stddef.h>

#include "types/charType.h"
#include "diagLog/msgKind.h"

//================================================================
//
// DiagLogThreading
//
//================================================================

struct DiagLogThreading
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

//----------------------------------------------------------------

struct DiagLogThreadingNull : public DiagLogThreading
{
    bool isThreadProtected() const override
        {return true;}

    void lock() override
        {}

    void unlock() override
        {}
};

//================================================================
//
// DiagLogOutput
//
// The simplest diagnostic logging interface.
//
//================================================================

struct DiagLogOutput
{
    // Add message to the log.
    virtual bool addMsg(const CharType* msgStr, MsgKind msgKind) =0;

    // Clear log (if supported).
    virtual bool clear() =0;

    // Update log view (if supported).
    virtual bool update() =0;
};

//----------------------------------------------------------------

struct DiagLogOutputNull : public DiagLogOutput
{
    bool addMsg(const CharType* msgStr, MsgKind msgKind) override
        {return true;}

    bool clear() override
        {return true;}

    virtual bool update() override
        {return true;}
};

//================================================================
//
// DiagLog
//
//================================================================

struct DiagLog : public DiagLogThreading, public DiagLogOutput
{
};

//----------------------------------------------------------------

struct DiagLogNull : public DiagLogThreadingNull, DiagLogOutputNull
{
};
