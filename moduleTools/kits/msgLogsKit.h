#pragma once

#include "userOutput/msgLogKit.h"
#include "formatting/messageFormatterKit.h"

//================================================================
//
// LocalLogKit
// MsgLogsKit
//
// User message and error output kits.
//
//================================================================

KIT_CREATE(LocalLogKit, MsgLog&, localLog);
KIT_CREATE2(LocalLogAuxKit, bool, localLogAuxAvailable, MsgLog&, localLogAux);

using MsgLogsKit = KitCombine<MessageFormatterKit, MsgLogKit, LocalLogKit, LocalLogAuxKit>;
