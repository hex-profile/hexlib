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

KIT_COMBINE4(DiagnosticKit, MessageFormatterKit, MsgLogKit, ErrorLogKit, ErrorLogExKit);
