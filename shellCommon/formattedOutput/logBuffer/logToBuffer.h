#pragma once

#include "userOutput/msgLog.h"
#include "formattedOutput/logBuffer/logBuffer.h"
#include "formatting/messageFormatter.h"

//================================================================
//
// LogToBufferThunk
//
//================================================================

class LogToBufferThunk : public MsgLog
{

public:

    inline void setup(LogBufferWriting* outputInterface, MessageFormatter* formatter, Timer* timer)
    {
        this->outputInterface = outputInterface; 
        this->formatter = formatter;
        this->timer = timer;
    }

    inline void reset()
    {
        this->outputInterface = nullptr; 
        this->formatter = nullptr;
        this->timer = nullptr;
    }

public:

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind);

    bool clear()
    {
        if (outputInterface)
            ensure(outputInterface->clear());

        return true;
    }

    bool update()
        {return true;}

    bool isThreadProtected() const 
        {return false;}

    void lock()
        {}

    void unlock()
        {}

private:

    LogBufferWriting* outputInterface = nullptr;
    MessageFormatter* formatter = nullptr;
    Timer* timer = nullptr;

};
