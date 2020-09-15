#pragma once

#include <mutex>

#include "userOutput/msgLog.h"

//================================================================
//
// logToStdConsole
//
//================================================================

class logToStdConsole : public MsgLog
{

public:

    logToStdConsole(bool useDebugOutput, bool useStdErr)
        : 
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

    bool const useDebugOutput;
    bool const useStdErr;

};
