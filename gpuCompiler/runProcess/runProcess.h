#pragma once

#include "userOutput/errorLogExKit.h"
#include "stlString/stlString.h"
#include "stdFunc/stdFunc.h"

//================================================================
//
// RunProcessKit
//
//================================================================

KIT_COMBINE3(RunProcessKit, ErrorLogKit, MsgLogKit, ErrorLogExKit);

//================================================================
//
// runProcess
//
//================================================================

stdbool runProcess(const StlString& cmdLine, stdPars(RunProcessKit));
