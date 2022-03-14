#include "msgLogToBridge.h"

#include "compileTools/blockExceptionsSilent.h"

//================================================================
//
// toMessageKind
//
//================================================================

inline auto toMessageKind(MsgKind value)
{
    using debugBridge::MessageKind;

    COMPILE_ASSERT(int(MessageKind::Info) == int(msgInfo));
    COMPILE_ASSERT(int(MessageKind::Warning) == int(msgWarn));
    COMPILE_ASSERT(int(MessageKind::Error) == int(msgErr));

    return debugBridge::MessageKind(value);
}

//================================================================
//
// MsgLogToDiagLogAndBridge::addMsg
//
//================================================================

bool MsgLogToDiagLogAndBridge::addMsg(const FormatOutputAtom& v, MsgKind msgKind)
{
    formatter.clear();
    v.func(v.value, formatter);
    ensure(formatter.valid());

    ////

    bool diagOk = true;
    
    {
        diagLog.addMsg(formatter.cstr(), msgKind);

        if (flushEveryMessage)
            diagOk = diagOk && diagLog.update();
    }

    ////

    auto bridgeCode = [&] ()
    {
        bridgeLog.add(formatter.cstr(), toMessageKind(msgKind));
    };

    bool bridgeOk = true;

    if (bridgeActive)
        bridgeOk = blockExceptionsSilentVoid(bridgeCode());

    ////

    return diagOk && bridgeOk;
}
