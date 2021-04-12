#include "baseInterfaces/baseSignals.h"
#include "debugBridge/bridge/debugBridge.h"
#include "errorLog/blockExceptions.h"
#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"

//================================================================
//
// BaseActionSetupToBridge
//
//================================================================

class BaseActionSetupToBridge : public BaseActionSetup
{

public:

    using Kit = ErrorLogExKit;

    BaseActionSetupToBridge(debugBridge::ActionSetup& base, stdPars(Kit))
        : base(base), kit(kit), trace(trace) {}

    virtual bool actsetClear()
        {return blockExceptionsVoid(base.clear());}

    virtual bool actsetUpdate()
        {return blockExceptionsVoid(base.update());}

    virtual bool actsetAdd(BaseActionId id, const CharType* key, const CharType* name, const CharType* comment)
        {return blockExceptionsVoid(base.add({id, key, name, comment}));}

private:

    debugBridge::ActionSetup& base;
    Kit kit;
    TraceScope trace;

};
