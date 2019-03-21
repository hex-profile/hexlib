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
        require(rowOrg == 0 && rowEnd == LogBufferEnd);
        require(logA.readRange(receiver, 0, LogBufferEnd));
        require(logB.readRange(receiver, 0, LogBufferEnd));
        return true;
    }

public:

    inline LogBufferReadingCombinedThunk(LogBufferReading& logA, LogBufferReading& logB)
        : logA(logA), logB(logB) {}

private:

    LogBufferReading& logA;
    LogBufferReading& logB;

};
