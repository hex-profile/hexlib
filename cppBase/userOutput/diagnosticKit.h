#pragma once

#include "userOutput/errorLogExKit.h"
#include "errorLog/errorLogKit.h"
#include "userOutput/msgLogKit.h"
#include "formatting/messageFormatterKit.h"

//================================================================
//
// DiagnosticKit
//
//================================================================

using DiagnosticKit = KitCombine<MessageFormatterKit, MsgLogKit, ErrorLogKit, ErrorLogExKit>;
