#pragma once

#include "baseInterfaces/actionDefs.h"
#include "charType/charArray.h"
#include "cfg/cfgSerialization.h"
#include "data/array.h"
#include "stdFunc/stdFunc.h"
#include "storage/adapters/callable.h"
#include "userOutput/msgLogExKit.h"

namespace signalTools {

//================================================================
//
// Kit
//
//================================================================

using Kit = MsgLogExKit;

//================================================================
//
// gatherActionSet
//
// Also sets signal IDs.
//
//================================================================

using ActionReceiver = Callable<void (ActionId id, CharArray name, CharArray key, CharArray comment)>;

stdbool gatherActionSet(CfgSerialization& serialization, ActionReceiver& receiver, size_t& actionCount, stdPars(Kit));

//================================================================
//
// updateSignals
//
// Sets all signal counts to zero and increases counts
// for those signals that received actions.
//
//================================================================

using ActionIdReceiver = Callable<void (ActionId id)>;
using ActionIdProvider = Callable<void (ActionIdReceiver& receiver)>;

void updateSignals(bool providerHasData, ActionIdProvider& provider, CfgSerialization& serialization, const Array<int32>& actionHist);

//----------------------------------------------------------------

}
