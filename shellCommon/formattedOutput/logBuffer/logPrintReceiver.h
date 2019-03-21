#pragma once

#include "compileTools/classContext.h"
#include "userOutput/msgLog.h"
#include "formattedOutput/logBuffer/logBuffer.h"

//================================================================
//
// LogPrintReceiver
//
//================================================================

class LogPrintReceiver : public LogBufferReceiver
{

public:

    bool addRow(const CharArray& text, MsgKind kind, const TimeMoment& moment)
    {
        bool ok = msgLog.addMsg(FormatOutputAtom(text), kind);
        check_flag(ok, allOk);
        return ok;
    }

public:

    bool allOk = true;

private:

    CLASS_CONTEXT(LogPrintReceiver, ((MsgLog&, msgLog)))

};
