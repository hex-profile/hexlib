#pragma once

#include "interfaces/syncObjects.h"
#include "formattedOutput/logBuffer/logBuffer.h"
#include "storage/rememberCleanup.h"

//================================================================
//
// LogBufferMtThunk
//
//================================================================

#define GUARD_IF_LOCK_CREATED \
    if (lock.created()) lock.lock(); \
    REMEMBER_CLEANUP1(if (lock.created()) lock.unlock(), Mutex&, lock)

//----------------------------------------------------------------

class LogBufferMtThunk: public LogBufferIO
{

public:

    bool add(const CharArray& text, MsgKind kind, const TimeMoment& moment)
    {
        GUARD_IF_LOCK_CREATED;

        ensure(baseWriter.add(text, kind, moment));
        return true;
    }

    bool clear()
    {
        GUARD_IF_LOCK_CREATED;

        ensure(baseWriter.clear());
        return true;
    }

public:

    bool readRange(LogBufferReceiver& receiver, RowInt rowOrg, RowInt rowEnd)
    {
        GUARD_IF_LOCK_CREATED;

        ensure(baseReader.readRange(receiver, rowOrg, rowEnd));
        return true;
    }

public:

    bool multithreaded() const
        {return lock.created();}

    Mutex& getLock()
        {return lock;}

public:

    LogBufferMtThunk(Mutex& lock, LogBufferWriting& baseWriter, LogBufferReading& baseReader)
        : lock(lock), baseWriter(baseWriter), baseReader(baseReader) {}

private:

    Mutex& lock;

    LogBufferWriting& baseWriter;
    LogBufferReading& baseReader;

};

////

#undef GUARD_IF_LOCK_CREATED
