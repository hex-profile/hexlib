#pragma once

#include "userOutput/msgLog.h"
#include "channels/buffers/logBuffer/logBuffer.h"
#include "formatting/messageFormatter.h"
#include "lib/logToBuffer/debuggerOutputControl.h"
#include "numbers/int/intType.h"

//================================================================
//
// LogUpdater
//
//================================================================

using LogUpdater = Callable<void ()>;

//================================================================
//
// LogToBufferContext
//
//================================================================

struct LogToBufferContext
{
    LogBufferWriting& logBuffer;
    MessageFormatter& formatter;
    Timer& timer;
    int debugOutputLevel;
    LogUpdater& logUpdater;
};

//================================================================
//
// LogToBufferThunk
//
//================================================================

struct LogToBufferThunk
    :
    public MsgLog,
    public DebuggerOutputControl,
    private LogToBufferContext
{
    LogToBufferThunk(const LogToBufferContext& context)
        : LogToBufferContext{context} {}

    ////

    virtual void setDebuggerOutputLevel(int level)
    {
        debugOutputLevel = level;
    }

    ////

    virtual bool addMsg(const FormatOutputAtom& v, MsgKind msgKind);

    virtual bool clear()
    {
        logBuffer.clearLog();
        return true;
    }

    virtual bool update()
    {
        logUpdater();
        return true;
    }

    virtual void lock()
    {
    }

    virtual void unlock()
    {
    }
};
