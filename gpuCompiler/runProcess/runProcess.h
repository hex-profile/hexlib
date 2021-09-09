#pragma once

#include "userOutput/errorLogExKit.h"
#include "stlString/stlString.h"
#include "stdFunc/stdFunc.h"

//================================================================
//
// RunProcessKit
//
//================================================================

using RunProcessKit = KitCombine<ErrorLogKit, MsgLogKit, ErrorLogExKit>;

//================================================================
//
// runProcess
//
//================================================================

stdbool runProcess(const StlString& cmdLine, stdPars(RunProcessKit));
