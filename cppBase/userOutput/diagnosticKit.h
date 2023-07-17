#pragma once

#include "userOutput/msgLogExKit.h"
#include "errorLog/errorLogKit.h"
#include "userOutput/msgLogKit.h"
#include "formatting/messageFormatterKit.h"

//================================================================
//
// DiagnosticKit
//
//================================================================

using DiagnosticKit = KitCombine<MessageFormatterKit, MsgLogKit, ErrorLogKit, MsgLogExKit>;
