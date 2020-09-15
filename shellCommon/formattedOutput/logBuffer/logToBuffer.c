#include "logToBuffer.h"

#include "stdFunc/stdFunc.h"
#include "formattedOutput/formatStreamStdio.h"

//================================================================
//
// LogToBufferThunk::addMsg
//
//================================================================

bool LogToBufferThunk::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    constexpr size_t bufferSize = 1024;
    CharType bufferArray[bufferSize];
    FormatStreamStdioThunk formatter{bufferArray, bufferSize};

    v.func(v.value, formatter);
    ensure(formatter.valid());

    if (outputInterface && timer)
        outputInterface->add(CharArray(formatter.data(), formatter.size()), msgKind, timer->moment());

    return true;
}
