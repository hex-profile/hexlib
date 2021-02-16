#include "diagLogTool.h"

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
