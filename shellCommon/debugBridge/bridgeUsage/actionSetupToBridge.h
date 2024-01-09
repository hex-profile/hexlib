#include "baseInterfaces/baseSignals.h"
#include "debugBridge/bridge/debugBridge.h"
#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"
#include "errorLog/convertExceptions.h"

//================================================================
//
// BaseActionSetupToBridge
//
//================================================================

class BaseActionSetupToBridge : public BaseActionSetup
{

public:

    using Kit = MsgLogExKit;

    BaseActionSetupToBridge(debugBridge::ActionSetup& base, stdPars(Kit))
        : base(base), stdParsCapture {}

    virtual bool actsetClear()
    {
        convertExceptionsBegin
        {
            base.clear();
        }
        convertExceptionsEndEx(return false)
        return true;
    }

    virtual bool actsetAdd(ActionId id, ActionKey key, const CharType* name, const CharType* comment)
    {
        convertExceptionsBegin
        {
            debugBridge::ActionParamsRef action{id, key, name, comment};
            base.add({&action, 1});
        }
        convertExceptionsEndEx(return false)
        return true;
    }

    virtual bool actsetUpdate()
    {
        return true;
    }

private:

    debugBridge::ActionSetup& base;
    stdParsMember(Kit);

};
