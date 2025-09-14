#pragma once

#include "userOutput/msgLogExKit.h"
#include "stlString/stlString.h"
#include "stdFunc/stdFunc.h"

//================================================================
//
// RunProcessKit
//
//================================================================

using RunProcessKit = KitCombine<ErrorLogKit, MsgLogKit, MsgLogExKit>;

//================================================================
//
// runProcess
//
//================================================================

void runProcess(const StlString& cmdLine, stdPars(RunProcessKit));
