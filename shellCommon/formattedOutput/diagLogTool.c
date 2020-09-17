#include "diagLogTool.h"

#include "stdFunc/stdFunc.h"
#include "formattedOutput/messageFormatterStdio.h"
#include "stlString/stlString.h"

//================================================================
//
// MsgLogByDiagLog::addMsg
//
//================================================================

bool MsgLogByDiagLog::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    formatter.clear();
    v.func(v.value, formatter);
    ensure(formatter.valid());

    return base.addMsg(formatter.data(), msgKind);
}
