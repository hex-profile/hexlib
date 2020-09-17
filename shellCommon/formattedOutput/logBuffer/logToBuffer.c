#include "logToBuffer.h"

#include "stdFunc/stdFunc.h"
#include "formattedOutput/messageFormatterStdio.h"

//================================================================
//
// LogToBufferThunk::addMsg
//
//================================================================

bool LogToBufferThunk::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    ensure(outputInterface && formatter && timer);

    formatter->clear();
    v.func(v.value, *formatter);
    ensure(formatter->valid());

    outputInterface->add(formatter->charArray(), msgKind, timer->moment());

    return true;
}
