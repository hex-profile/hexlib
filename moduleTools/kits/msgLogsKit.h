#pragma once

#include "userOutput/msgLogKit.h"

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

KIT_COMBINE3(MsgLogsKit, MsgLogKit, LocalLogKit, LocalLogAuxKit);
