#pragma once

#include "formattedOutput/logBuffer/logBuffer.h"

//================================================================
//
// LogBufferDebugOutputThunk
//
//================================================================

class LogBufferDebugOutputThunk : public LogBufferWriting
{

public:

    bool add(const CharArray& text, MsgKind kind, const TimeMoment& moment);

    bool clear()
        {return baseBuffer.clear();}

public:

    LogBufferDebugOutputThunk(LogBufferWriting& baseBuffer, bool enabled)
        : baseBuffer(baseBuffer), enabled(enabled) {}

private:

    bool const enabled;
    LogBufferWriting& baseBuffer;

};
