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

    bool addMsg(const FormatOutputAtom& v, MsgKind msgKind);

    bool clear()
        {return true;}

    bool update()
        {return true;}

    void lock()
        {mutex.lock();}

    void unlock()
        {mutex.unlock();}

private:

    std::recursive_mutex mutex;

    MessageFormatter& formatter;

    bool const useDebugOutput;
    bool const useStdErr;

};
