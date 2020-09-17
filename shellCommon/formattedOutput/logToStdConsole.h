#pragma once

#include <mutex>

#include "userOutput/msgLog.h"
#include "formatting/messageFormatter.h"

//================================================================
//
// LogToStdConsole
//
//================================================================

class LogToStdConsole : public MsgLog
{

public:

    LogToStdConsole(MessageFormatter& formatter, bool useDebugOutput, bool useStdErr)
        : 
        formatter(formatter),
        useDebugOutput(useDebugOutput),
        useStdErr(useStdErr)
    {
    }

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind) override;

    bool clear() override
        {return true;}

    bool update() override
        {return true;}

    bool isThreadProtected() const override
        {return true;}

    void lock() override
        {mutex.lock();}

    void unlock() override
        {mutex.unlock();}

private:

    std::recursive_mutex mutex;

    MessageFormatter& formatter;

    bool const useDebugOutput;
    bool const useStdErr;

};
