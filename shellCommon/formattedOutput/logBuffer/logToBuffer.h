#pragma once

#include "userOutput/msgLog.h"

#include "formattedOutput/logBuffer/logBuffer.h"

//================================================================
//
// LogToBufferThunk
//
//================================================================

class LogToBufferThunk : public MsgLog
{

public:

    inline void setup(LogBufferWriting* outputInterface, Timer* timer)
        {this->outputInterface = outputInterface; this->timer = timer;}

    inline void reset()
        {this->outputInterface = 0; this->timer = 0;}

public:

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind);

    bool clear()
    {
        if (outputInterface)
            require(outputInterface->clear());

        return true;
    }

    bool update()
        {return true;}

private:

    LogBufferWriting* outputInterface = 0;
    Timer* timer = 0;

};
