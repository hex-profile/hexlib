#pragma once

#include "formattedOutput/logBuffer/logBuffer.h"

//================================================================
//
// LogBufferReadingCombinedThunk
//
//================================================================

class LogBufferReadingCombinedThunk : public LogBufferReading
{

public:

    bool readRange(LogBufferReceiver& receiver, RowInt rowOrg, RowInt rowEnd)
    {
        ensure(rowOrg == 0 && rowEnd == LogBufferEnd);
        ensure(logA.readRange(receiver, 0, LogBufferEnd));
        ensure(logB.readRange(receiver, 0, LogBufferEnd));
        return true;
    }

public:

    inline LogBufferReadingCombinedThunk(LogBufferReading& logA, LogBufferReading& logB)
        : logA(logA), logB(logB) {}

private:

    LogBufferReading& logA;
    LogBufferReading& logB;

};
