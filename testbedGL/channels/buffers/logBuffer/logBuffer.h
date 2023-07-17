#pragma once

#include "charType/charArray.h"
#include "extLib/userOutput/msgKind.h"
#include "storage/adapters/callable.h"
#include "storage/optionalObject.h"
#include "storage/smartPtr.h"
#include "timer/timer.h"

//================================================================
//
// LogBufferConfig
//
//================================================================

struct LogBufferConfig
{
    virtual void setHistoryLimit(size_t value) =0;
};

//================================================================
//
// LogBufferWriting
//
// Write a message with timestamp to log buffer.
//
//================================================================

struct LogBufferWriting
{
    virtual void addMessage(const CharArray& text, MsgKind kind, const TimeMoment& moment) =0;

    virtual void clearLog() =0;

    virtual void refreshAllMoments(const TimeMoment& moment) =0;

    virtual size_t messageCount() const =0;
};

//================================================================
//
// LogBufferReceiver
//
//================================================================

using LogBufferReceiver = Callable<void (const CharArray& text, MsgKind kind, const TimeMoment& moment)>;

//================================================================
//
// LogBufferReading
//
//================================================================

struct LogBufferReading
{
    virtual OptionalObject<TimeMoment> getLastModification() =0;

    // Reads the last N rows (or less if there is not enough data).
    virtual void readLastMessages(LogBufferReceiver& receiver, size_t count) =0;

    // Reads the first N rows (or less if there is not enough data).
    // If the log cannot be displayed fully, warns about it in the last row.
    virtual void readFirstMessagesShowOverflow(LogBufferReceiver& receiver, size_t count) =0;
};

//================================================================
//
// LogBuffer
//
//================================================================

struct LogBuffer : public LogBufferConfig, public LogBufferReading, public LogBufferWriting
{
    static UniquePtr<LogBuffer> create();
    virtual ~LogBuffer() {}

    virtual void clearMemory() =0;

    //
    // Buffer API.
    //

    virtual bool hasUpdates() const =0;

    virtual void reset() =0;

    virtual bool absorb(LogBuffer& other) =0;

    virtual void moveFrom(LogBuffer& other) =0;
};
