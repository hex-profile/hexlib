#include "logToBuffer.h"

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

//================================================================
//
// LogToBufferThunk::addMsg
//
//================================================================

bool LogToBufferThunk::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    auto moment = timer.moment();

    ////

    formatter.clear();
    v.func(v.value, formatter);
    formatter.write("\n", 1);
    ensure(formatter.valid());

    ////

    if (msgKind >= debugOutputLevel)
    {
    #if defined(_WIN32)
        OutputDebugString(formatter.cstr());
    #endif
    }

    ////

    auto text = formatter.str();
    if (text.size) text.size -= 1; // Remove '\n'

    logBuffer.addMessage(text, msgKind, moment);

    return true;
}
