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
    if (lock.created()) lock->enter(); \
    REMEMBER_CLEANUP1(if (lock.created()) lock->leave(), CriticalSection&, lock)

//----------------------------------------------------------------

class LogBufferMtThunk: public LogBufferIO
{

public:

    bool add(const CharArray& text, MsgKind kind, const TimeMoment& moment)
    {
        GUARD_IF_LOCK_CREATED;

        require(baseWriter.add(text, kind, moment));
        return true;
    }

    bool clear()
    {
        GUARD_IF_LOCK_CREATED;

        require(baseWriter.clear());
        return true;
    }

public:

    bool readRange(LogBufferReceiver& receiver, RowInt rowOrg, RowInt rowEnd)
    {
        GUARD_IF_LOCK_CREATED;

        require(baseReader.readRange(receiver, rowOrg, rowEnd));
        return true;
    }

public:

    bool multithreaded() const
        {return lock.created();}

    CriticalSection& getLock()
        {return lock;}

public:

    LogBufferMtThunk(CriticalSection& lock, LogBufferWriting& baseWriter, LogBufferReading& baseReader)
        : lock(lock), baseWriter(baseWriter), baseReader(baseReader) {}

private:

    CriticalSection& lock;

    LogBufferWriting& baseWriter;
    LogBufferReading& baseReader;

};

////

#undef GUARD_IF_LOCK_CREATED
