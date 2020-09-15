#include "diagLogTool.h"

#include "stdFunc/stdFunc.h"
#include "formattedOutput/formatStreamStdio.h"
#include "stlString/stlString.h"

//================================================================
//
// MsgLogByDiagLog::addMsg
//
//================================================================

bool MsgLogByDiagLog::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    using namespace std;

    if (!output)
        return true;

    ////

    constexpr size_t bufferSize = 1024;
    CharType bufferArray[bufferSize];
    FormatStreamStdioThunk formatter{bufferArray, bufferSize};

    v.func(v.value, formatter);
    ensure(formatter.valid());

    output->addMsg(formatter.data(), msgKind);

    ////

    return true;
}
