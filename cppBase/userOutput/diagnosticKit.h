#pragma once

#include "userOutput/errorLogExKit.h"
#include "errorLog/errorLogKit.h"
#include "userOutput/msgLogKit.h"

//================================================================
//
// DiagnosticKit
//
//================================================================

KIT_COMBINE3(DiagnosticKit, ErrorLogKit, MsgLogKit, ErrorLogExKit);
