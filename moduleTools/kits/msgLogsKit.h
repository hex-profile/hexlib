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

KIT_CREATE1(LocalLogKit, MsgLog&, localLog);

KIT_CREATE2(LocalLogAuxKit, bool, localLogAuxAvailable, MsgLog&, localLogAux);

KIT_COMBINE4(MsgLogsKit, MessageFormatterKit, MsgLogKit, LocalLogKit, LocalLogAuxKit);
