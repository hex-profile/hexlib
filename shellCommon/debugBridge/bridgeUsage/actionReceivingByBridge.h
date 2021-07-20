#include "baseInterfaces/baseSignals.h"
#include "debugBridge/bridge/debugBridge.h"
#include "errorLog/blockExceptions.h"
#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"

namespace actionReceivingByBridge {

namespace db = debugBridge;

//================================================================
//
// Check predefined IDs.
//
//================================================================

COMPILE_ASSERT(baseActionId::MouseLeftDown == db::actionId::MouseLeftDown);
COMPILE_ASSERT(baseActionId::MouseLeftUp == db::actionId::MouseLeftUp);

COMPILE_ASSERT(baseActionId::MouseRightDown == db::actionId::MouseRightDown);
COMPILE_ASSERT(baseActionId::MouseRightUp == db::actionId::MouseRightUp);

COMPILE_ASSERT(baseActionId::WheelDown == db::actionId::WheelDown);
COMPILE_ASSERT(baseActionId::WheelUp == db::actionId::WheelUp);

COMPILE_ASSERT(baseActionId::SaveConfig == db::actionId::SaveConfig);
COMPILE_ASSERT(baseActionId::LoadConfig == db::actionId::LoadConfig);
COMPILE_ASSERT(baseActionId::EditConfig == db::actionId::EditConfig);

COMPILE_ASSERT(baseActionId::ResetupActions == db::actionId::ResetupActions);

//================================================================
//
// BridgeReceiverByBaseReceiver
//
//================================================================

class BridgeReceiverByBaseReceiver : public db::ActionReceiver
{

public:

    BridgeReceiverByBaseReceiver(BaseActionReceiver& receiver)
        : receiver{receiver} {}

    virtual void process(db::ArrayRef<const db::ActionRec> actions)
    {
        auto arr = makeArray(actions.ptr, actions.size);
        receiver.process(recastElement<const BaseActionRec>(arr));
    }

private:

    BaseActionReceiver& receiver;

};

//================================================================
//
// BaseActionReceivingByBridge
//
//================================================================

class BaseActionReceivingByBridge : public BaseActionReceiving
{

public:

    using Kit = ErrorLogExKit;

    BaseActionReceivingByBridge(db::ActionReceiving& base, stdPars(Kit))
        : base(base), kit(kit), trace(trace) {}

    virtual void getActions(BaseActionReceiver& receiver)
    {
        BridgeReceiverByBaseReceiver receiverThunk{receiver};
        blockExceptionsVoid(base.getActions(receiverThunk));
    }

private:

    db::ActionReceiving& base;
    Kit kit;
    TraceScope trace;

};

//----------------------------------------------------------------

}

using actionReceivingByBridge::BaseActionReceivingByBridge;
