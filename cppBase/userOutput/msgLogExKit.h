#pragma once

#include "errorLog/errorLogKit.h"
#include "userOutput/msgLogKit.h"

//================================================================
//
// MsgLogExKit
//
// Message log with stack trace.
//
//================================================================

struct MsgLogEx;

KIT_CREATE(MsgLogExKit, MsgLogEx&, msgLogEx);
